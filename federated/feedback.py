"""
``/feedback`` — ground-truth reporting contract (brief §2.3).

A bank's ops team closes the loop: "this L2 hold was a confirmed scam", "this
L3 lock was a false positive". That label, joined to the feature vector the
bank scored at decision time, is what ``LocalTrainer.learn`` consumes.

Boundary rules
--------------
* The handler runs **inside the bank's environment**, next to its
  ``LocalTrainer``. Feature vectors are looked up from the bank's own decision
  log (``FeatureLog``); they never travel with the feedback message and the
  central aggregator never sees this endpoint at all.
* Poisoning defence lives in two places: (1) the aggregator's coordinate-median
  rule (§5.4) bounds what a compromised bank can do to *everyone else*; (2)
  ``FeedbackHandler`` bounds what a compromised ops account can do to *this*
  bank — one label per transaction, only for transactions this bank actually
  scored, only within a reporting window, and every accepted event is
  auditable.

Wiring: this module is framework-free. Mount ``FeedbackHandler.handle`` behind
whatever HTTP layer the deployment uses (n8n webhook node, FastAPI, ...); the
request body is ``FeedbackEvent`` and the response is ``FeedbackResult``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from adapters.schema import Identifier

from .local_trainer import LocalTrainer


class Outcome(str, Enum):
    CONFIRMED_FRAUD = "confirmed_fraud"  # label 1
    FALSE_POSITIVE = "false_positive"  # label 0 — we actioned it, it was legit
    CONFIRMED_LEGIT = "confirmed_legit"  # label 0 — customer-verified or aged out clean

    @property
    def label(self) -> int:
        return 1 if self is Outcome.CONFIRMED_FRAUD else 0


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank_id: Identifier
    transaction_id: Identifier
    outcome: Outcome
    reported_by: Identifier = Field(description="ops user / system id, for audit — not a free-text field")
    reported_at_utc: datetime
    #: Optional case reference from the bank's own fraud-ops system.
    case_ref: Optional[Identifier] = None


class FeedbackResult(BaseModel):
    accepted: bool
    reason: str
    transaction_id: str
    label_applied: Optional[int] = None


class FeatureLog:
    """The bank's decision-time record: transaction_id → (feature vector, scored_at).

    In production this is the bank's own datastore. Kept in-memory here; the
    interface is the point. Nothing in it ever leaves the bank.
    """

    def __init__(self):
        self._rows: dict[str, tuple[np.ndarray, datetime]] = {}

    def record(self, transaction_id: str, x: np.ndarray, scored_at: datetime) -> None:
        self._rows[transaction_id] = (np.asarray(x, dtype=float).copy(), scored_at)

    def get(self, transaction_id: str) -> Optional[tuple[np.ndarray, datetime]]:
        return self._rows.get(transaction_id)


class FeedbackHandler:
    def __init__(
        self,
        trainer: LocalTrainer,
        feature_log: FeatureLog,
        reporting_window: timedelta = timedelta(days=90),
        epochs_per_event: int = 1,
    ):
        self.trainer = trainer
        self.log = feature_log
        self.window = reporting_window
        self.epochs = epochs_per_event
        self._labelled: dict[str, Outcome] = {}
        self.audit: list[FeedbackEvent] = []

    def handle(self, ev: FeedbackEvent) -> FeedbackResult:
        if ev.bank_id.lower() != self.trainer.bank_id.lower():
            return FeedbackResult(accepted=False, reason="bank_id does not match this trainer", transaction_id=ev.transaction_id)
        if ev.transaction_id in self._labelled:
            return FeedbackResult(accepted=False, reason="transaction already labelled; relabelling is not permitted via this endpoint", transaction_id=ev.transaction_id)
        found = self.log.get(ev.transaction_id)
        if found is None:
            return FeedbackResult(accepted=False, reason="transaction was never scored by this bank's model", transaction_id=ev.transaction_id)
        x, scored_at = found
        reported = ev.reported_at_utc if ev.reported_at_utc.tzinfo else ev.reported_at_utc.replace(tzinfo=timezone.utc)
        if reported < scored_at:
            return FeedbackResult(accepted=False, reason="feedback timestamp precedes the scoring event", transaction_id=ev.transaction_id)
        if reported - scored_at > self.window:
            return FeedbackResult(accepted=False, reason=f"outside the {self.window.days}-day reporting window", transaction_id=ev.transaction_id)

        y = ev.outcome.label
        self.trainer.learn(x, np.array([y]), epochs=self.epochs)
        self._labelled[ev.transaction_id] = ev.outcome
        self.audit.append(ev)
        return FeedbackResult(accepted=True, reason="ok", transaction_id=ev.transaction_id, label_applied=y)
