"""
TCS BaNCS adapter — ILLUSTRATIVE, UNVERIFIED.

==============================================================================
  WARNING — THIS FIELD MAPPING IS NOT VALIDATED AGAINST A REAL BaNCS
  INTEGRATION. It is modelled on the publicly visible style of TCS BaNCS
  digital/REST interfaces (PascalCase keys, ``TxnDetails`` grouping, integer
  amounts in paise, epoch-millisecond timestamps, 1/0 flags, DOB rather than
  age) purely to DEMONSTRATE a third payload shape plugging into the same
  registry. Historically many BaNCS deployments expose SOAP/XML rather than
  JSON; a real integration might need an XML→dict step before ``normalize``.
  Rewrite against the integrating bank's API document before any live use.
==============================================================================

Auth scheme assumed here: a static per-bank API key in ``X-TCS-API-Key``,
compared constant-time against the key pinned in the ``SecretProvider``.
Deliberately the weakest of the three example schemes, to make the point that
the adapter layer *reports* what a bank's auth is, it doesn't make it stronger:
a bank offering only a static key should be flagged at onboarding review.

Mapping behaviours exercised here that the other two adapters don't:

* **Unit conversion.** ``AmountInPaise`` (integer) → ``amount_inr`` (Decimal
  with 2 dp). Done with Decimal arithmetic, never float division.
* **Age from DOB.** ``SenderDOB`` → ``sender_age`` computed at transaction time.
* **Coded tri-state.** ``VoiceLivenessCode`` 0/1/2 → None/True/False.
"""

from __future__ import annotations

import hmac
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping

from .base_adapter import IST, BankAdapter, NormalizationError, header_lookup
from .schema import KavachTransaction, PaymentRail
from .secrets import SecretProvider

# Illustrative channel codes → rail. UNVERIFIED.
_BANCS_RAIL_MAP: dict[str, PaymentRail] = {
    "01": PaymentRail.UPI,
    "02": PaymentRail.IMPS,
    "03": PaymentRail.NEFT,
    "04": PaymentRail.RTGS,
    "05": PaymentRail.CARD,
    "09": PaymentRail.INTRA_BANK,
}

# Illustrative liveness codes → tri-state. 0 = not checked / unavailable.
_LIVENESS_CODE: dict[int, bool | None] = {0: None, 1: True, 2: False}


class BancsAdapter(BankAdapter):
    """Adapter for a bank whose core runs TCS BaNCS (illustrative)."""

    core_banking_system = "TCS BaNCS (illustrative mapping)"
    SECRET_NAME = "api_key"

    def __init__(self, bank_id: str, secrets: SecretProvider):
        self.bank_id = bank_id
        self._secrets = secrets

    # ------------------------------------------------------------------ auth
    def validate_auth(self, headers: Mapping[str, str], body: bytes = b"") -> bool:
        key = self._secrets.get(self.bank_id, self.SECRET_NAME)
        if not key:
            return False  # fail closed
        presented = header_lookup(headers, "X-TCS-API-Key")
        if not presented:
            return False
        return hmac.compare_digest(presented.strip().encode(), key.strip())

    # ------------------------------------------------------------- normalize
    def normalize(self, raw_payload: Mapping[str, Any]) -> KavachTransaction:
        # Illustrative BaNCS-style payload:
        # {
        #   "TxnDetails": {"TxnRefNo": "BNC2026091100077", "TxnTimestamp": 1757579002000,
        #                  "AmountInPaise": 15000000, "CurrencyCode": "INR",
        #                  "ChannelCode": "01", "PayerRemarks": "..."},
        #   "PayerDetails": {"SenderDOB": "1959-02-14", "AcctOpenedDaysAgo": 4120},
        #   "PayeeDetails": {"PayeeName": "...", "PayeeAcctAgeDays": 3, "IsNewPayee": 1},
        #   "DeviceContext": {"OnCall": 1, "VoiceLivenessCode": 0, "GeoLabel": "Patna, BR"}
        # }
        txn = self._section(raw_payload, "TxnDetails")
        payer = self._section(raw_payload, "PayerDetails")
        payee = self._section(raw_payload, "PayeeDetails")
        dev = self._section(raw_payload, "DeviceContext")

        ts = txn.require_datetime("TxnTimestamp", fmt=None, assume_tz=IST)  # epoch ms → UTC

        currency = str(txn.optional("CurrencyCode", "INR")).upper()
        if currency != "INR":
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"Kavach only handles INR; got currency '{currency}'",
                fields=[txn.path("CurrencyCode")],
            )

        paise = txn.require_int("AmountInPaise")
        amount_inr = (Decimal(paise) / Decimal(100)).quantize(Decimal("0.01"))

        code = txn.optional("ChannelCode")
        rail = _BANCS_RAIL_MAP.get(str(code).zfill(2), PaymentRail.UNKNOWN) if code is not None else PaymentRail.UNKNOWN

        liveness_code = dev.optional_int("VoiceLivenessCode")
        if liveness_code is not None and liveness_code not in _LIVENESS_CODE:
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"unrecognised VoiceLivenessCode {liveness_code}",
                fields=[dev.path("VoiceLivenessCode")],
            )

        mapped: dict[str, Any] = {
            "transaction_id": txn.require("TxnRefNo"),
            "bank_id": self.bank_id,
            "timestamp_utc": ts,
            "amount_inr": amount_inr,
            "rail": rail,
            "sender_age": _age_from_dob(self, payer.require("SenderDOB"), ts, payer.path("SenderDOB")),
            "sender_account_age_days": payer.optional_int("AcctOpenedDaysAgo"),
            "receiver_account_age_days": payee.require_int("PayeeAcctAgeDays"),
            "receiver_name": payee.optional("PayeeName"),
            "is_new_beneficiary": payee.optional_tristate("IsNewPayee"),
            "is_active_phone_call": dev.require_bool("OnCall"),
            "caller_verified_biometric": None if liveness_code is None else _LIVENESS_CODE[liveness_code],
            "device_location": dev.require("GeoLabel"),
            "remarks": txn.optional("PayerRemarks"),
        }
        return self._build(mapped)


# ---------------------------------------------------------------- utilities
def _age_from_dob(adapter: BankAdapter, dob: Any, at: datetime, field_name: str) -> int:
    try:
        d = date.fromisoformat(str(dob))
    except ValueError as e:
        raise NormalizationError(
            bank_id=adapter.bank_id,
            code="bad_value",
            message=f"could not parse date of birth '{dob}'",
            fields=[field_name],
        ) from e
    today = at.astimezone(IST).date()
    years = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return years
