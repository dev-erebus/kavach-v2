"""
Finacle (Infosys) adapter — ILLUSTRATIVE, UNVERIFIED.

==============================================================================
  WARNING — THIS FIELD MAPPING IS NOT VALIDATED AGAINST A REAL FINACLE
  INTEGRATION. It is modelled on publicly documented Finacle conventions
  (Finacle Integrator / FI-XML style message envelopes, uppercase snake-case
  field names, 'Y'/'N' flags, 'DD-MM-YYYY HH:MM:SS' IST timestamps) as a
  DEMONSTRATION of the adapter pattern. Every bank running Finacle customises
  its schema; SBI's payload will not equal PNB's. Before any live deployment
  this file must be rewritten against the specific bank's API contract and
  signed off by that bank's integration team.
==============================================================================

Auth scheme assumed here: HMAC-SHA256 over the raw request body, hex-encoded,
presented in ``X-Finacle-Signature``. Again — illustrative. Real Finacle
deployments commonly sit behind an API gateway (mTLS + OAuth2 client
credentials); the adapter would then check gateway-injected identity headers
instead.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base_adapter import IST, BankAdapter, NormalizationError, header_lookup
from .schema import KavachTransaction, PaymentRail
from .secrets import SecretProvider

# Illustrative mapping of Finacle transaction-type codes → Kavach rails.
# UNVERIFIED: real Finacle TRAN_TYPE / TRAN_SUB_TYPE codes are bank-specific.
_FINACLE_RAIL_MAP: dict[str, PaymentRail] = {
    "UPI": PaymentRail.UPI,
    "IMPS": PaymentRail.IMPS,
    "NEFT": PaymentRail.NEFT,
    "RTGS": PaymentRail.RTGS,
    "POS": PaymentRail.CARD,
    "ECOM": PaymentRail.CARD,
    "TFR": PaymentRail.INTRA_BANK,
}


class FinacleAdapter(BankAdapter):
    """Adapter for a bank whose core runs Infosys Finacle (illustrative)."""

    core_banking_system = "Infosys Finacle (illustrative mapping)"

    #: Name under which this adapter asks the SecretProvider for its key.
    SECRET_NAME = "hmac_secret"

    def __init__(self, bank_id: str, secrets: SecretProvider):
        # Multiple banks run Finacle, so bank_id is injected rather than fixed,
        # and the signing key is looked up PER BANK — SBI and PNB never share one.
        self.bank_id = bank_id
        self._secrets = secrets

    # ------------------------------------------------------------------ auth
    def validate_auth(self, headers: Mapping[str, str], body: bytes = b"") -> bool:
        secret = self._secrets.get(self.bank_id, self.SECRET_NAME)
        if not secret:
            # Fail closed: a bank with no secret configured gets nothing accepted.
            return False
        presented = header_lookup(headers, "X-Finacle-Signature")
        return self._hmac_matches(secret, body, presented or "")

    # ------------------------------------------------------------- normalize
    def normalize(self, raw_payload: Mapping[str, Any]) -> KavachTransaction:
        # Illustrative Finacle envelope:
        # {
        #   "FIXML": {
        #     "Header": {"BankId": "...", "MsgId": "..."},
        #     "Body": {
        #       "TranDtls": {
        #         "TRAN_ID": "...", "TRAN_DATE": "11-09-2026 14:03:22",
        #         "TRAN_AMT": "150000.00", "TRAN_CRNCY_CODE": "INR",
        #         "TRAN_TYPE": "UPI",
        #         "TRAN_RMKS": "..."
        #       },
        #       "CustDtls":  {"CUST_AGE": "67", "ACCT_OPN_DAYS": "4120"},
        #       "BenefDtls": {"BENEF_NAME": "...", "BENEF_ACCT_OPN_DAYS": "3",
        #                     "FIRST_TIME_BENEF": "Y"},
        #       "ChnlCtx":   {"ACTIVE_CALL_FLG": "Y", "VOICE_LIVENESS": "UNKNOWN",
        #                     "DEVICE_GEO": "Lucknow, UP"}
        #     }
        #   }
        # }
        tran = self._section(raw_payload, "FIXML", "Body", "TranDtls")
        cust = self._section(raw_payload, "FIXML", "Body", "CustDtls")
        benef = self._section(raw_payload, "FIXML", "Body", "BenefDtls")
        chnl = self._section(raw_payload, "FIXML", "Body", "ChnlCtx")

        currency = str(tran.optional("TRAN_CRNCY_CODE", "INR")).upper()
        if currency != "INR":
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"Kavach only handles INR; got currency '{currency}'",
                fields=[tran.path("TRAN_CRNCY_CODE")],
            )

        rail = _FINACLE_RAIL_MAP.get(str(tran.optional("TRAN_TYPE", "")).upper(), PaymentRail.UNKNOWN)

        mapped: dict[str, Any] = {
            "transaction_id": tran.require("TRAN_ID"),
            "bank_id": self.bank_id,
            "timestamp_utc": tran.require_datetime("TRAN_DATE", fmt="%d-%m-%Y %H:%M:%S", assume_tz=IST),
            "amount_inr": str(tran.require("TRAN_AMT")),
            "rail": rail,
            "sender_age": cust.require_int("CUST_AGE"),
            "sender_account_age_days": cust.optional_int("ACCT_OPN_DAYS"),
            "receiver_account_age_days": benef.require_int("BENEF_ACCT_OPN_DAYS"),
            "receiver_name": benef.optional("BENEF_NAME"),
            "is_new_beneficiary": benef.optional_tristate("FIRST_TIME_BENEF"),
            "is_active_phone_call": chnl.require_bool("ACTIVE_CALL_FLG"),
            # Tri-state on purpose: absent / "UNKNOWN" -> None, never False (brief §5.1).
            "caller_verified_biometric": chnl.optional_tristate("VOICE_LIVENESS"),
            "device_location": chnl.require("DEVICE_GEO"),
            "remarks": tran.optional("TRAN_RMKS"),
        }
        return self._build(mapped)

