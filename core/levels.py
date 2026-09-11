"""Kavach threat levels. Shared by rules (v1), threshold bridge (v2) and the eval harness."""

from __future__ import annotations

from enum import IntEnum


class ThreatLevel(IntEnum):
    """Ordinal — higher is more severe. Comparisons like ``level >= L2`` are meaningful."""

    L0 = 0  # allow
    L1 = 1  # allow + in-app warning / cooling-off nudge
    L2 = 2  # hold + step-up verification (OTP on a *different* channel, callback)
    L3 = 3  # hard lock + human review + forensic dossier (+ FR-2 report, Component C)

    @property
    def action(self) -> str:
        return _ACTIONS[self]


_ACTIONS = {
    ThreatLevel.L0: "allow",
    ThreatLevel.L1: "warn",
    ThreatLevel.L2: "hold_and_verify",
    ThreatLevel.L3: "hard_lock_and_review",
}
