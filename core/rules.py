"""
v1 rule-based classifier — RECONSTRUCTED, see note below.

==============================================================================
  NOTE ON PROVENANCE. The v1 thresholds live in an n8n Switch node whose
  workflow export was not available when this module was written. The
  ``RuleConfig`` defaults below are a reconstruction from the v2 build brief
  (₹1,00,000 as the headline anomalous amount; elderly senders, fresh receiver
  accounts and an active phone call as the L2/L3 signals). They are meant to
  be REPLACED by the exported v1 numbers — change ``RuleConfig`` defaults, or
  pass a config, and nothing else moves. Until then, eval numbers produced
  with this module describe *a* v1-style rule pipeline, not *the* deployed one.
==============================================================================

Why the rules are written as an explicit, ordered decision list rather than a
score: it mirrors how an n8n Switch node evaluates (first matching branch
wins), so the port back into the live workflow is mechanical, and so the eval
harness can attribute every decision to exactly one named rule.

``classify()`` returns the level and the name of the rule that fired, so the
eval harness can report *why* transactions landed where they did — the
false-positive breakdown the brief asks for is impossible without this.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import NamedTuple

from adapters.schema import KavachTransaction

from .levels import ThreatLevel


@dataclass(frozen=True)
class RuleConfig:
    """Every hardcoded number in the v1 logic, in one place.

    RECONSTRUCTED DEFAULTS — replace with the exported n8n Switch-node values.
    """

    # Amount bands (INR). ``high`` is the brief's headline ₹1,00,000.
    amount_high_inr: Decimal = Decimal("100000")
    amount_medium_inr: Decimal = Decimal("50000")
    amount_low_inr: Decimal = Decimal("10000")

    # Victim-profile signal: elderly sender.
    elderly_age: int = 60

    # Mule-profile signal: beneficiary account younger than this many days.
    fresh_receiver_days: int = 7
    young_receiver_days: int = 30


class RuleDecision(NamedTuple):
    level: ThreatLevel
    rule: str  # name of the branch that fired, e.g. "L3_high_amount_call_elderly"


class V1RuleClassifier:
    """Ordered decision list. First matching branch wins, like an n8n Switch node."""

    def __init__(self, config: RuleConfig | None = None):
        self.cfg = config or RuleConfig()

    # The rule table is data, not code paths, so the eval harness can list it.
    RULE_NAMES: tuple[str, ...] = (
        "L3_high_amount_call_and_profile",
        "L2_high_amount_alone",
        "L2_call_and_fresh_receiver",
        "L2_medium_amount_and_signal",
        "L1_low_amount",
        "L1_call_alone",
        "L1_young_receiver",
        "L0_default",
    )

    def classify(self, tx: KavachTransaction) -> RuleDecision:
        c = self.cfg
        amt = tx.amount_inr
        call = tx.is_active_phone_call
        elderly = tx.sender_age >= c.elderly_age
        fresh_rx = tx.receiver_account_age_days <= c.fresh_receiver_days
        young_rx = tx.receiver_account_age_days <= c.young_receiver_days
        # §5.1 / §5.5: an *unknown* biometric verdict is not a signal either
        # way. Only an explicit ``False`` (liveness check failed) escalates.
        cloned_voice = tx.caller_verified_biometric is False

        # ---- L3: hard lock -------------------------------------------------
        if amt >= c.amount_high_inr and call and (elderly or fresh_rx or cloned_voice):
            return RuleDecision(ThreatLevel.L3, "L3_high_amount_call_and_profile")

        # ---- L2: hold + verify ---------------------------------------------
        if amt >= c.amount_high_inr:
            return RuleDecision(ThreatLevel.L2, "L2_high_amount_alone")
        if call and (fresh_rx or cloned_voice):
            return RuleDecision(ThreatLevel.L2, "L2_call_and_fresh_receiver")
        if amt >= c.amount_medium_inr and (call or elderly or fresh_rx):
            return RuleDecision(ThreatLevel.L2, "L2_medium_amount_and_signal")

        # ---- L1: warn ------------------------------------------------------
        if amt >= c.amount_low_inr:
            return RuleDecision(ThreatLevel.L1, "L1_low_amount")
        if call:
            return RuleDecision(ThreatLevel.L1, "L1_call_alone")
        if young_rx:
            return RuleDecision(ThreatLevel.L1, "L1_young_receiver")

        return RuleDecision(ThreatLevel.L0, "L0_default")
