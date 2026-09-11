"""
Canonical Kavach transaction schema.

Every bank adapter's ``normalize()`` must return a ``KavachTransaction``. This is
the ONLY shape that Nodes 3–10 of the n8n pipeline (LLM classification, rule
switch, freeze/lock actions, dossier) are allowed to see. Nothing bank-specific
survives past this boundary.

Design notes
------------
* ``extra="forbid"`` — an adapter that leaks an unmapped bank field is a bug,
  and we want it to fail loudly at the boundary rather than have the field
  silently reach the LLM prompt.
* Free-text fields (``device_location``, ``receiver_name``, ``remarks``) are
  typed as ``FreeText`` so they can be identified and sanitized before prompt
  interpolation (see §5.2 of the build brief). Sanitization itself lives in
  ``/hardening``; the schema only enforces a hard length cap so a payload can't
  smuggle a multi-kilobyte prompt in a location string.
* ``caller_verified_biometric`` is tri-state (§5.1). ``None`` means "unknown /
  no liveness provider available" and MUST be treated as unknown downstream —
  never coerced to ``False`` or ``True``.
* ``remarks`` is the **primary prompt-injection surface**. On UPI the payer
  types it directly into the app, so it is the most attacker-controlled string
  in the whole schema. It never enters the feature vector, and the §5.2
  sanitizer treats it as the main threat: heavily escaped, length-clamped, and
  passed to the LLM only inside a delimited data block, if at all.
* ``FEATURE_CONTRACT`` (bottom of file) fixes the exact ordered feature vector
  every bank's local model consumes. Optional fields get a value slot AND a
  missing-indicator slot, so a bank whose core system never sends
  ``is_new_beneficiary`` still emits the same-shaped weight vector as one that
  does. Without this, FedAvg (Component B) would average misaligned
  coordinates silently. The contract lives HERE, next to the fields, so the
  adapters and the learner cannot drift apart.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, ClassVar, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Hard cap on any free-text field. Long enough for a real city/branch string,
# short enough that it can't carry a paragraph of injected instructions.
FREE_TEXT_MAX_LEN = 200

FreeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=FREE_TEXT_MAX_LEN),
]

# Bank / transaction identifiers: printable ASCII only, bounded length.
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._:\-]+$",
    ),
]


class PaymentRail(str, Enum):
    """Indian retail payment rails Kavach cares about."""

    UPI = "UPI"
    IMPS = "IMPS"
    NEFT = "NEFT"
    RTGS = "RTGS"
    CARD = "CARD"
    INTRA_BANK = "INTRA_BANK"
    UNKNOWN = "UNKNOWN"


class KavachTransaction(BaseModel):
    """Canonical, validated transaction record consumed by the Kavach pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    # --- identity -----------------------------------------------------------
    transaction_id: Identifier = Field(
        description="Bank's own transaction reference. Unique per bank_id, not globally."
    )
    bank_id: Identifier = Field(
        description="Kavach-side bank identifier, e.g. 'sbi', 'hdfc'. Matches registry key."
    )
    timestamp_utc: datetime = Field(
        description="Transaction initiation time, normalized to UTC."
    )

    # --- money --------------------------------------------------------------
    amount_inr: Decimal = Field(
        description="Transaction amount in INR. Decimal, not float — money.",
    )
    rail: PaymentRail = Field(default=PaymentRail.UNKNOWN)

    # --- sender context (the potential victim) -----------------------------
    sender_age: int = Field(
        ge=18,
        le=120,
        description="Sender's age in years. Kavach's L2/L3 logic is elderly-weighted.",
    )
    sender_account_age_days: Optional[int] = Field(
        default=None, ge=0, description="How long the sender account has existed."
    )

    # --- receiver context (the potential mule) ------------------------------
    receiver_account_age_days: int = Field(
        ge=0,
        description="Age of the beneficiary account in days. Fresh accounts are a mule signal.",
    )
    receiver_name: Optional[FreeText] = Field(
        default=None, description="Beneficiary display name. FREE TEXT — sanitize before LLM."
    )
    is_new_beneficiary: Optional[bool] = Field(
        default=None,
        description="True if the sender has never paid this receiver before. None if bank can't say.",
    )

    # --- social-engineering signals ----------------------------------------
    is_active_phone_call: bool = Field(
        description="Was the sender on a phone call at initiation time (device-side signal)."
    )
    caller_verified_biometric: Optional[bool] = Field(
        default=None,
        description=(
            "Result of a third-party voice-liveness check on the active call. "
            "None = unknown / no provider. NEVER coerce None to a boolean."
        ),
    )
    device_location: FreeText = Field(
        description="Device geo string as reported by the bank app. FREE TEXT — sanitize before LLM."
    )
    remarks: Optional[FreeText] = Field(
        default=None,
        description="Payer-entered transaction note / UPI remark. FREE TEXT — sanitize before LLM.",
    )

    # --- validators ---------------------------------------------------------
    @field_validator("amount_inr", mode="before")
    @classmethod
    def _coerce_amount(cls, v):
        # Accept int/str/Decimal; reject float explicitly so adapters don't
        # accidentally pass binary-float rupee values through.
        if isinstance(v, float):
            raise ValueError("amount_inr must not be a float; pass str, int or Decimal")
        if isinstance(v, (int, str)):
            v = Decimal(str(v))
        return v

    @field_validator("amount_inr")
    @classmethod
    def _amount_positive(cls, v: Decimal) -> Decimal:
        if not v.is_finite():
            raise ValueError("amount_inr must be finite")
        if v <= 0:
            raise ValueError("amount_inr must be > 0")
        # Paise precision at most.
        if v.as_tuple().exponent < -2:
            raise ValueError("amount_inr must have at most 2 decimal places")
        return v

    @field_validator("timestamp_utc")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "timestamp_utc must be timezone-aware; adapters must attach the bank's tz"
            )
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _biometric_requires_call(self):
        # A liveness verdict without a call is a mapping bug in the adapter.
        if self.caller_verified_biometric is not None and not self.is_active_phone_call:
            raise ValueError(
                "caller_verified_biometric set but is_active_phone_call is False"
            )
        return self

    # --- helpers ------------------------------------------------------------
    #: Every free-text field, most attacker-controlled first. ``remarks`` is
    #: payer-typed on UPI; ``receiver_name`` is chosen by whoever opened the mule
    #: account; ``device_location`` comes from the bank app but may be spoofed.
    FREE_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("remarks", "receiver_name", "device_location")

    def free_text_values(self) -> dict[str, str]:
        """Return every populated free-text field. Used by the §5.2 sanitizer."""
        return {
            name: getattr(self, name)
            for name in self.FREE_TEXT_FIELDS
            if getattr(self, name) is not None
        }

    def to_feature_vector(self) -> list[float]:
        """Emit the fixed-shape numeric feature vector defined by ``FEATURE_CONTRACT``."""
        return FEATURE_CONTRACT.vectorize(self)


# --------------------------------------------------------------------------- #
# Feature contract — READ BEFORE TOUCHING /federated
# --------------------------------------------------------------------------- #
class FeatureContract:
    """The single source of truth for what a Kavach model's input vector looks like.

    Rules, in priority order:

    1. **Shape is fixed.** ``dim`` is a constant for a given ``version``. Every
       bank, every adapter, every learner produces exactly ``dim`` floats in
       exactly this order. Model store refuses weights whose length or contract
       version differ.
    2. **Optional fields take two slots.** A value slot, imputed with
       ``IMPUTE_VALUE`` when the field is ``None``, and a missing-indicator slot
       that is ``1.0`` when the field is ``None`` else ``0.0``. The indicator
       lets the model learn that "bank X never sends this" is itself
       information, and makes the imputation constant irrelevant to fit.
    3. **Transforms are fixed here, never fitted per bank.** A per-bank
       StandardScaler would give each bank a different coordinate system and
       break averaging as surely as a shape mismatch would. So: ``log1p`` for
       heavy-tailed magnitudes, a fixed divisor for ages, one-hot for the rail
       enum. The learner may add a bias term; it may not rescale inputs.
    4. **Free text never enters the vector.** ``remarks``, ``receiver_name`` and
       ``device_location`` are LLM-side context only (after §5.2 sanitization).
       Nothing attacker-typed should be able to move a federated weight.
    5. **Changing anything above bumps ``version``** and invalidates every
       stored model. That is the intended cost.
    """

    version: int = 1
    IMPUTE_VALUE: float = 0.0
    AGE_SCALE: float = 100.0  # sender_age / 100 -> roughly [0.18, 1.2]

    #: Ordered slot names. ``*_missing`` slots are the indicators from rule 2.
    names: tuple[str, ...] = (
        # required
        "log1p_amount_inr",
        "sender_age_scaled",
        "log1p_receiver_account_age_days",
        "is_active_phone_call",
        # optional: value + missing indicator
        "log1p_sender_account_age_days",
        "sender_account_age_days_missing",
        "is_new_beneficiary",
        "is_new_beneficiary_missing",
        "caller_verified_biometric",
        "caller_verified_biometric_missing",
        # rail one-hot, in PaymentRail declaration order
        *(f"rail_{r.value.lower()}" for r in PaymentRail),
    )

    @property
    def dim(self) -> int:
        return len(self.names)

    def vectorize(self, tx: "KavachTransaction") -> list[float]:
        def opt_num(v: Optional[int]) -> tuple[float, float]:
            return (self.IMPUTE_VALUE, 1.0) if v is None else (math.log1p(float(v)), 0.0)

        def opt_bool(v: Optional[bool]) -> tuple[float, float]:
            return (self.IMPUTE_VALUE, 1.0) if v is None else (float(v), 0.0)

        vec: list[float] = [
            math.log1p(float(tx.amount_inr)),
            tx.sender_age / self.AGE_SCALE,
            math.log1p(float(tx.receiver_account_age_days)),
            float(tx.is_active_phone_call),
            *opt_num(tx.sender_account_age_days),
            *opt_bool(tx.is_new_beneficiary),
            *opt_bool(tx.caller_verified_biometric),
            *(1.0 if tx.rail is r else 0.0 for r in PaymentRail),
        ]
        assert len(vec) == self.dim, "FeatureContract.names and vectorize() are out of sync"
        return vec


FEATURE_CONTRACT = FeatureContract()
