"""
Temenos T24 / Transact adapter — ILLUSTRATIVE, UNVERIFIED.

==============================================================================
  WARNING — THIS FIELD MAPPING IS NOT VALIDATED AGAINST A REAL TEMENOS
  INTEGRATION. It is modelled on the publicly visible style of Temenos'
  REST APIs (camelCase JSON, ``header``/``body`` envelope, ISO-8601 timestamps
  with offset, ``debitParty``/``creditParty`` naming, dates rather than
  precomputed ages) purely to DEMONSTRATE that a second vendor with a
  different shape plugs into the same registry with zero pipeline changes.
  Every T24 deployment is heavily customised; HDFC's contract will differ from
  Axis's. Rewrite against the real bank API document before any live use.
==============================================================================

Auth scheme assumed here: mutual TLS terminated at the bank's API gateway,
which injects the SHA-256 fingerprint of the presented client certificate as
``X-Client-Cert-Fingerprint``. The adapter compares it (constant-time) against
the fingerprint pinned for this bank in the ``SecretProvider``. This is a
common gateway pattern, chosen so the three example adapters exercise three
different auth styles (HMAC / mTLS-fingerprint / API key). Illustrative.

Two mapping behaviours worth noticing, because real integrations hit them:

* **Derived fields.** T24-style payloads carry account *open dates*, not ages.
  The adapter computes ``*_account_age_days`` relative to the transaction
  timestamp — so the canonical field is derived, not copied, and a bad date
  produces a structured ``bad_value`` error rather than a wrong age.
* **JSON numbers.** Amounts arrive as JSON numbers. ``registry.ingest`` parses
  with ``parse_float=Decimal``; if this adapter is handed a raw ``float``
  (someone bypassed ingest) the schema rejects it loudly. That is deliberate.
"""

from __future__ import annotations

import hmac
from datetime import date, datetime
from typing import Any, Mapping

from .base_adapter import IST, BankAdapter, NormalizationError, header_lookup
from .schema import KavachTransaction, PaymentRail
from .secrets import SecretProvider

# Illustrative paymentType → rail. UNVERIFIED against any T24 product catalogue.
_T24_RAIL_MAP: dict[str, PaymentRail] = {
    "UPI": PaymentRail.UPI,
    "IMPS": PaymentRail.IMPS,
    "NEFT": PaymentRail.NEFT,
    "RTGS": PaymentRail.RTGS,
    "CARD": PaymentRail.CARD,
    "AC.TRANSFER": PaymentRail.INTRA_BANK,
    "FUNDS.TRANSFER": PaymentRail.INTRA_BANK,
}

# Illustrative liveness status vocabulary → tri-state.
_BIOMETRIC_STATUS: dict[str, bool | None] = {
    "VERIFIED": True,
    "FAILED": False,
    "NOT_AVAILABLE": None,
    "NOT_CHECKED": None,
    "UNKNOWN": None,
}


class TemenosAdapter(BankAdapter):
    """Adapter for a bank whose core runs Temenos T24 / Transact (illustrative)."""

    core_banking_system = "Temenos T24 / Transact (illustrative mapping)"
    SECRET_NAME = "mtls_cert_fingerprint"

    def __init__(self, bank_id: str, secrets: SecretProvider):
        self.bank_id = bank_id
        self._secrets = secrets

    # ------------------------------------------------------------------ auth
    def validate_auth(self, headers: Mapping[str, str], body: bytes = b"") -> bool:
        pinned = self._secrets.get(self.bank_id, self.SECRET_NAME)
        if not pinned:
            return False  # fail closed
        presented = header_lookup(headers, "X-Client-Cert-Fingerprint")
        if not presented:
            return False
        norm = presented.replace(":", "").strip().lower().encode()
        return hmac.compare_digest(norm, pinned.replace(b":", b"").strip().lower())

    # ------------------------------------------------------------- normalize
    def normalize(self, raw_payload: Mapping[str, Any]) -> KavachTransaction:
        # Illustrative T24-style envelope:
        # {
        #   "header": {"transactionId": "FT26254ABCDE",
        #              "transactionDateTime": "2026-09-11T14:03:22+05:30"},
        #   "body": {
        #     "transaction": {"amount": 150000.00, "currency": "INR",
        #                     "paymentType": "IMPS", "narrative": "..."},
        #     "debitParty":  {"customerAge": 67, "accountOpenDate": "2015-03-01"},
        #     "creditParty": {"name": "...", "accountOpenDate": "2026-09-08",
        #                     "existingBeneficiary": false},
        #     "channelContext": {"activeVoiceCall": true,
        #                        "callerBiometricStatus": "NOT_AVAILABLE",
        #                        "deviceLocation": "Mumbai, MH"}
        #   }
        # }
        hdr = self._section(raw_payload, "header")
        txn = self._section(raw_payload, "body", "transaction")
        debit = self._section(raw_payload, "body", "debitParty")
        credit = self._section(raw_payload, "body", "creditParty")
        chnl = self._section(raw_payload, "body", "channelContext")

        # ISO-8601; if the offset is missing we assume IST (bank-local).
        ts = hdr.require_datetime("transactionDateTime", fmt=None, assume_tz=IST)

        currency = str(txn.optional("currency", "INR")).upper()
        if currency != "INR":
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"Kavach only handles INR; got currency '{currency}'",
                fields=[txn.path("currency")],
            )

        raw_status = str(chnl.optional("callerBiometricStatus", "UNKNOWN")).upper()
        if raw_status not in _BIOMETRIC_STATUS:
            # An unrecognised status is NOT "unknown" — it's a mapping gap. Fail.
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"unrecognised callerBiometricStatus '{raw_status}'",
                fields=[chnl.path("callerBiometricStatus")],
            )

        existing = credit.optional("existingBeneficiary")
        is_new = None if existing is None else (not self._parse_bool(existing, credit.path("existingBeneficiary")))

        mapped: dict[str, Any] = {
            "transaction_id": hdr.require("transactionId"),
            "bank_id": self.bank_id,
            "timestamp_utc": ts,
            "amount_inr": _money(self, txn.require("amount"), txn.path("amount")),
            "rail": _T24_RAIL_MAP.get(str(txn.optional("paymentType", "")).upper(), PaymentRail.UNKNOWN),
            "sender_age": debit.require_int("customerAge"),
            "sender_account_age_days": _age_days(self, debit.optional("accountOpenDate"), ts, debit.path("accountOpenDate")),
            "receiver_account_age_days": _age_days(self, credit.require("accountOpenDate"), ts, credit.path("accountOpenDate")),
            "receiver_name": credit.optional("name"),
            "is_new_beneficiary": is_new,
            "is_active_phone_call": chnl.require_bool("activeVoiceCall"),
            "caller_verified_biometric": _BIOMETRIC_STATUS[raw_status],
            "device_location": chnl.require("deviceLocation"),
            "remarks": txn.optional("narrative"),
        }
        return self._build(mapped)


# ---------------------------------------------------------------- utilities
def _money(adapter: BankAdapter, value: Any, field_name: str) -> Any:
    """Pass Decimal/int/str through; refuse binary floats with a pointer to the fix."""
    if isinstance(value, float):
        raise NormalizationError(
            bank_id=adapter.bank_id,
            code="bad_value",
            message="amount arrived as a binary float; parse JSON with parse_float=Decimal (registry.ingest does)",
            fields=[field_name],
        )
    return str(value)


def _age_days(adapter: BankAdapter, open_date: Any, at: datetime, field_name: str) -> int | None:
    """Days between an ISO date and the transaction time. None passes through as None."""
    if open_date is None:
        return None
    try:
        d = date.fromisoformat(str(open_date))
    except ValueError as e:
        raise NormalizationError(
            bank_id=adapter.bank_id,
            code="bad_value",
            message=f"could not parse date '{open_date}'",
            fields=[field_name],
        ) from e
    days = (at.astimezone(IST).date() - d).days
    if days < 0:
        raise NormalizationError(
            bank_id=adapter.bank_id,
            code="bad_value",
            message=f"account open date {d} is after the transaction date",
            fields=[field_name],
        )
    return days
