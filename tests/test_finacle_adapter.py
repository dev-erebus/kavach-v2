import hmac, hashlib, json
from decimal import Decimal
import pytest

from adapters.finacle_adapter import FinacleAdapter
from adapters.base_adapter import NormalizationError
from adapters.schema import PaymentRail
from adapters.secrets import StaticSecretProvider

SECRET = b"test-secret"
SECRETS = StaticSecretProvider({("sbi", "hmac_secret"): SECRET})

def sbi():
    return FinacleAdapter("sbi", SECRETS)

def payload(**overrides):
    p = {
        "FIXML": {
            "Header": {"BankId": "SBIN", "MsgId": "m1"},
            "Body": {
                "TranDtls": {"TRAN_ID": "SBI-20260911-000123", "TRAN_DATE": "11-09-2026 14:03:22",
                             "TRAN_AMT": "150000.00", "TRAN_CRNCY_CODE": "INR", "TRAN_TYPE": "UPI",
                             "TRAN_RMKS": "electricity bill urgent"},
                "CustDtls": {"CUST_AGE": "67", "ACCT_OPN_DAYS": "4120"},
                "BenefDtls": {"BENEF_NAME": "R KUMAR", "BENEF_ACCT_OPN_DAYS": "3", "FIRST_TIME_BENEF": "Y"},
                "ChnlCtx": {"ACTIVE_CALL_FLG": "Y", "VOICE_LIVENESS": "UNKNOWN", "DEVICE_GEO": "Lucknow, UP"},
            },
        }
    }
    for path, val in overrides.items():
        cur = p["FIXML"]["Body"]
        *parents, leaf = path.split(".")
        for k in parents: cur = cur[k]
        if val is ...: del cur[leaf]
        else: cur[leaf] = val
    return p

def test_happy_path():
    tx = sbi().normalize(payload())
    assert tx.bank_id == "sbi"
    assert tx.amount_inr == Decimal("150000.00")
    assert tx.rail is PaymentRail.UPI
    assert tx.sender_age == 67
    assert tx.receiver_account_age_days == 3
    assert tx.is_new_beneficiary is True
    assert tx.is_active_phone_call is True
    assert tx.caller_verified_biometric is None          # UNKNOWN must stay None
    assert tx.timestamp_utc.isoformat() == "2026-09-11T08:33:22+00:00"  # IST -> UTC
    assert tx.free_text_values() == {"device_location": "Lucknow, UP",
                                     "receiver_name": "R KUMAR",
                                     "remarks": "electricity bill urgent"}

def test_missing_required_field_is_structured():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"ChnlCtx.DEVICE_GEO": ...}))
    err = ei.value
    assert err.code == "missing_field"
    assert err.fields == ["FIXML.Body.ChnlCtx.DEVICE_GEO"]
    assert err.to_dict()["bank_id"] == "sbi"

def test_schema_violation_underage_sender():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"CustDtls.CUST_AGE": "15"}))
    assert ei.value.code == "schema_violation"
    assert ei.value.fields == ["sender_age"]

def test_bad_amount_rejected():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"TranDtls.TRAN_AMT": "-5"}))
    assert ei.value.code == "schema_violation"

def test_non_inr_rejected():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"TranDtls.TRAN_CRNCY_CODE": "USD"}))
    assert ei.value.code == "bad_value"

def test_bad_date_rejected():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"TranDtls.TRAN_DATE": "2026/09/11"}))
    assert ei.value.code == "bad_value"

def test_biometric_without_call_rejected():
    with pytest.raises(NormalizationError):
        sbi().normalize(
            payload(**{"ChnlCtx.ACTIVE_CALL_FLG": "N", "ChnlCtx.VOICE_LIVENESS": "Y"}))

def test_oversized_free_text_rejected():
    with pytest.raises(NormalizationError) as ei:
        sbi().normalize(payload(**{"ChnlCtx.DEVICE_GEO": "x" * 500}))
    assert "device_location" in ei.value.fields

def test_auth():
    a = sbi()
    body = json.dumps(payload()).encode()
    sig = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    assert a.validate_auth({"x-finacle-signature": sig}, body) is True
    assert a.validate_auth({"X-Finacle-Signature": "deadbeef"}, body) is False
    assert a.validate_auth({}, body) is False
    # No secret configured for this bank -> fail closed, even with a "valid" sig
    assert FinacleAdapter("pnb", SECRETS).validate_auth({"X-Finacle-Signature": sig}, body) is False
    # Different bank, different key: a PNB-signed body must not verify as SBI
    pnb_sig = hmac.new(b"pnb-secret", body, hashlib.sha256).hexdigest()
    assert FinacleAdapter("sbi", SECRETS).validate_auth({"X-Finacle-Signature": pnb_sig}, body) is False
