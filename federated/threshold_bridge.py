"""
Threshold bridge: model probability → L0–L3, with per-bank calibrated cutoffs.

This replaces the hardcoded branches in n8n's Switch node (brief §2.3). The
cutoffs are not numbers someone typed; they are derived per bank from that
bank's own recent score distribution, so "₹1,00,000 is routine here / alarming
there" is expressed as "top 0.1% of *this* bank's risk scores", which is what
the L3 budget actually means operationally.

Calibration modes
-----------------
* ``calibrate_by_flag_rate(scores, targets)`` — unlabelled. Choose cutoffs so
  that a given share of traffic lands at ≥L1 / ≥L2 / ≥L3. This is what a bank
  can do on day one with no confirmed outcomes: it decides how much manual
  review it can afford and the bridge honours that budget.
* ``calibrate_by_precision(scores, labels, targets)`` — labelled. Choose the
  loosest cutoff at which precision at ≥L meets the target. Needs feedback
  history; used once ``/feedback`` has accumulated enough confirmed outcomes.

Failure posture (brief §5.5)
----------------------------
If the bridge has no model, no calibration, a spec mismatch, or a non-finite
score, it returns **L2 (hold and verify) with ``fallback=True``** — never L0.
An unverifiable signal degrades to human review, not to silent trust. The
caller can see ``fallback`` and route the transaction to the v1 rules as a
second opinion, but it cannot get an "allow" out of a broken bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np

from core.levels import ThreatLevel

from .feature_spec import FeatureSpec, SpecMismatch
from .local_trainer import OnlineLogisticRegression


@dataclass(frozen=True)
class Cutoffs:
    """Probability thresholds. ``p >= l3`` → L3, elif ``p >= l2`` → L2, elif ``p >= l1`` → L1, else L0."""

    l1: float
    l2: float
    l3: float

    def __post_init__(self):
        if not (0.0 <= self.l1 <= self.l2 <= self.l3 <= 1.0):
            raise ValueError(f"cutoffs must satisfy 0 <= l1 <= l2 <= l3 <= 1, got {self}")

    def level(self, p: float) -> ThreatLevel:
        if p >= self.l3:
            return ThreatLevel.L3
        if p >= self.l2:
            return ThreatLevel.L2
        if p >= self.l1:
            return ThreatLevel.L1
        return ThreatLevel.L0


@dataclass(frozen=True)
class FlagRateTargets:
    """Share of traffic a bank is willing to action at each tier. Ops budget, not a fraud estimate."""

    l1: float = 0.05  # 5% get a warning nudge
    l2: float = 0.01  # 1% get held for step-up verification
    l3: float = 0.001  # 0.1% get hard-locked and reviewed

    def __post_init__(self):
        if not (1.0 >= self.l1 >= self.l2 >= self.l3 >= 0.0):
            raise ValueError("flag-rate targets must be nested: l1 >= l2 >= l3")


class BridgeDecision(NamedTuple):
    level: ThreatLevel
    probability: Optional[float]
    fallback: bool
    reason: str


FALLBACK_LEVEL = ThreatLevel.L2  # hold + verify. A human looks. Never L0.


def calibrate_by_flag_rate(scores: np.ndarray, targets: FlagRateTargets = FlagRateTargets()) -> Cutoffs:
    """Unlabelled calibration: cutoffs = upper quantiles of this bank's own score distribution."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 100:
        raise ValueError(f"need at least 100 scores to calibrate, got {s.size}")
    q = lambda rate: float(np.quantile(s, 1.0 - rate)) if rate > 0 else 1.0  # noqa: E731
    l1, l2, l3 = q(targets.l1), q(targets.l2), q(targets.l3)
    # Ties at the top of a heavily skewed distribution can invert the order; enforce nesting.
    l2 = max(l2, l1)
    l3 = max(l3, l2)
    return Cutoffs(l1, l2, l3)


def calibrate_by_precision(
    scores: np.ndarray, labels: np.ndarray, precision_targets: tuple[float, float, float] = (0.02, 0.10, 0.50)
) -> Cutoffs:
    """Labelled calibration: loosest cutoff whose precision-at-or-above meets each tier's target.

    Falls back to the top score if no cutoff meets a target (then that tier is effectively off,
    which is the honest outcome — better than pretending a precision the data can't support).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape:
        raise ValueError("scores and labels must align")
    if y.sum() == 0:
        raise ValueError("cannot calibrate by precision with zero positive labels")
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    cum_tp = np.cumsum(y_sorted)
    prec = cum_tp / np.arange(1, len(s_sorted) + 1)

    def cutoff_for(target: float) -> float:
        ok = np.where(prec >= target)[0]
        if ok.size == 0:
            return float(s_sorted[0])  # nothing meets it; tier fires only at the very top
        return float(s_sorted[ok[-1]])  # loosest (largest index) position still meeting target

    l1, l2, l3 = (cutoff_for(t) for t in precision_targets)
    l2 = max(l2, l1)
    l3 = max(l3, l2)
    return Cutoffs(l1, l2, l3)


@dataclass
class ThresholdBridge:
    """Per-bank: a model + calibrated cutoffs → level. Fails to L2, never to L0."""

    bank_id: str
    spec: FeatureSpec
    model: Optional[OnlineLogisticRegression] = None
    cutoffs: Optional[Cutoffs] = None

    def score(self, x: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        try:
            p = float(self.model.predict_proba(np.asarray(x, dtype=float)[None, :])[0])
        except SpecMismatch:
            return None
        return p if np.isfinite(p) else None

    def decide(self, x: np.ndarray) -> BridgeDecision:
        if self.model is None:
            return BridgeDecision(FALLBACK_LEVEL, None, True, "no model loaded for this bank")
        if self.cutoffs is None:
            return BridgeDecision(FALLBACK_LEVEL, None, True, "bank cutoffs not calibrated")
        try:
            self.spec.check(np.asarray(x, dtype=float), "feature vector")
        except SpecMismatch as e:
            return BridgeDecision(FALLBACK_LEVEL, None, True, f"feature vector rejected: {e}")
        p = self.score(x)
        if p is None:
            return BridgeDecision(FALLBACK_LEVEL, None, True, "model produced no finite score")
        return BridgeDecision(self.cutoffs.level(p), p, False, "calibrated")

    def decide_many(self, X: np.ndarray) -> list[BridgeDecision]:
        return [self.decide(row) for row in np.asarray(X, dtype=float)]
