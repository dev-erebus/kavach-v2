"""Temenos + BaNCS adapters, registry ingestion, and the fixed feature contract."""
import hashlib
import hmac
import json
from decimal import Decimal

import pytest

from adapters.bancs_adapter import BancsAdapter
from adapters.base_adapter import AuthenticationError, NormalizationError
from adapters.finacle_adapter import FinacleAdapter
from adapters.registry import BANK_ID_HEADER, AdapterRegistry, default_registry
from adapters.schema import FEATURE_CONTRACT, KavachTransaction, PaymentRail
from adapters.secrets import EnvSecretProvider, StaticSecretProvider
from adapters.temenos_adapter import TemenosAdapter

FP = "AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89"
SECRETS = StaticSecretProvider({
    ("sbi", "hmac_secret"): b"sbi-secret",
    ("hdfc", "mtls_cert_fingerprint"): FP,
    ("demo_bancs_bank", "api_key"): "bancs-key-123",
})


# --------------------------------------------------------------------- payloads
def finacle_payload():
    return {"FIXML": {"Header": {"BankId": "SBIN"}, "Body": {
        "TranDtls": {"TRAN_ID": "SBI-1", "TRAN_DATE": "11-09-2026 14:03:22", "TRAN_AMT": "150000.00",
                     "TRAN_CRNCY_CODE": "INR", "TRAN_TYPE": "UPI", "TRAN_RMKS": "urgent"},
        "CustDtls": {"CUST_AGE": "67", "ACCT_OPN_DAYS": "4120"},
        "BenefDtls": {"BENEF_NAME": "R KUMAR", "BENEF_ACCT_OPN_DAYS": "3", "FIRST_TIME_BENEF": "Y"},
        "ChnlCtx": {"ACTIVE_CALL_FLG": "Y", "VOICE_LIVENESS": "UNKNOWN", "DEVICE_GEO": "Lucknow, UP"}}}}


def temenos_payload(**over):
    p = {"header": {"transactionId": "FT26254ABCDE", "transactionDateTime": "2026-09-11T14:03:22+05:30"},
         "body": {"transaction": {"amount": Decimal("150000.00"), "currency": "INR", "paymentType": "IMPS",
                                  "narrative": "urgent"},
                  "debitParty": {"customerAge": 67, "accountOpenDate": "2015-03-01"},
                  "creditParty": {"name": "R KUMAR", "accountOpenDate": "2026-09-08", "existingBeneficiary": False},
                  "channelContext": {"activeVoiceCall": True, "callerBiometricStatus": "NOT_AVAILABLE",
                                     "deviceLocation": "Mumbai, MH"}}}
    for k, v in over.items():
        sect, leaf = k.split(".")
        p["body"][sect][leaf] = v
    return p


def bancs_payload(**over):
    p = {"TxnDetails": {"TxnRefNo": "BNC2026091100077", "TxnTimestamp": 1789115602000, "AmountInPaise": 15000000,
                        "CurrencyCode": "INR", "ChannelCode": "01", "PayerRemarks": "urgent"},
         "PayerDetails": {"SenderDOB": "1959-02-14", "AcctOpenedDaysAgo": 4120},
         "PayeeDetails": {"PayeeName": "R KUMAR", "PayeeAcctAgeDays": 3, "IsNewPayee": 1},
         "DeviceContext": {"OnCall": 1, "VoiceLivenessCode": 0, "GeoLabel": "Patna, BR"}}
    for k, v in over.items():
        sect, leaf = k.split(".")
        p[sect][leaf] = v
    return p


# ---------------------------------------------------------------------- Temenos
def test_temenos_happy_path_derives_ages_from_dates():
    tx = TemenosAdapter("hdfc", SECRETS).normalize(temenos_payload())
    assert tx.amount_inr == Decimal("150000.00")
    assert tx.rail is PaymentRail.IMPS
    assert tx.receiver_account_age_days == 3            # 2026-09-08 -> 2026-09-11
    assert tx.sender_account_age_days == 4212           # 2015-03-01 -> 2026-09-11
    assert tx.is_new_beneficiary is True                 # existingBeneficiary=False -> new
    assert tx.caller_verified_biometric is None          # NOT_AVAILABLE -> None
    assert tx.timestamp_utc.isoformat() == "2026-09-11T08:33:22+00:00"


def test_temenos_rejects_raw_float_amount():
    with pytest.raises(NormalizationError) as ei:
        TemenosAdapter("hdfc", SECRETS).normalize(temenos_payload(**{"transaction.amount": 150000.0}))
    assert ei.value.code == "bad_value" and "parse_float" in ei.value.message


def test_temenos_unknown_biometric_status_is_error_not_unknown():
    with pytest.raises(NormalizationError) as ei:
        TemenosAdapter("hdfc", SECRETS).normalize(temenos_payload(**{"channelContext.callerBiometricStatus": "MAYBE"}))
    assert ei.value.fields == ["body.channelContext.callerBiometricStatus"]


def test_temenos_future_open_date_rejected():
    with pytest.raises(NormalizationError) as ei:
        TemenosAdapter("hdfc", SECRETS).normalize(temenos_payload(**{"creditParty.accountOpenDate": "2027-01-01"}))
    assert ei.value.code == "bad_value"


def test_temenos_auth_fingerprint():
    a = TemenosAdapter("hdfc", SECRETS)
    assert a.validate_auth({"X-Client-Cert-Fingerprint": FP}) is True
    assert a.validate_auth({"x-client-cert-fingerprint": FP.replace(":", "").lower()}) is True  # format-tolerant
    assert a.validate_auth({"X-Client-Cert-Fingerprint": FP[:-2] + "00"}) is False
    assert a.validate_auth({}) is False
    assert TemenosAdapter("axis", SECRETS).validate_auth({"X-Client-Cert-Fingerprint": FP}) is False  # not configured


# ------------------------------------------------------------------------ BaNCS
def test_bancs_happy_path_paise_and_dob():
    tx = BancsAdapter("demo_bancs_bank", SECRETS).normalize(bancs_payload())
    assert tx.amount_inr == Decimal("150000.00")         # 15000000 paise
    assert tx.rail is PaymentRail.UPI
    assert tx.sender_age == 67                           # DOB 1959-02-14 at 2026-09-11
    assert tx.is_new_beneficiary is True
    assert tx.is_active_phone_call is True
    assert tx.caller_verified_biometric is None          # code 0
    assert tx.timestamp_utc.isoformat() == "2026-09-11T08:33:22+00:00"


def test_bancs_liveness_codes():
    a = BancsAdapter("demo_bancs_bank", SECRETS)
    assert a.normalize(bancs_payload(**{"DeviceContext.VoiceLivenessCode": 1})).caller_verified_biometric is True
    assert a.normalize(bancs_payload(**{"DeviceContext.VoiceLivenessCode": 2})).caller_verified_biometric is False
    with pytest.raises(NormalizationError):
        a.normalize(bancs_payload(**{"DeviceContext.VoiceLivenessCode": 7}))


def test_bancs_odd_paise_keeps_precision():
    tx = BancsAdapter("demo_bancs_bank", SECRETS).normalize(bancs_payload(**{"TxnDetails.AmountInPaise": 1999}))
    assert tx.amount_inr == Decimal("19.99")


def test_bancs_auth_api_key():
    a = BancsAdapter("demo_bancs_bank", SECRETS)
    assert a.validate_auth({"X-TCS-API-Key": "bancs-key-123"}) is True
    assert a.validate_auth({"X-TCS-API-Key": "bancs-key-124"}) is False
    assert a.validate_auth({}) is False


# --------------------------------------------------------------------- registry
def test_registry_full_ingest_path():
    reg = AdapterRegistry([FinacleAdapter("sbi", SECRETS)])
    body = json.dumps(finacle_payload()).encode()
    sig = hmac.new(b"sbi-secret", body, hashlib.sha256).hexdigest()
    tx = reg.ingest({BANK_ID_HEADER: "sbi", "X-Finacle-Signature": sig}, body)
    assert isinstance(tx, KavachTransaction) and tx.bank_id == "sbi"


def test_registry_unknown_bank():
    reg = AdapterRegistry([FinacleAdapter("sbi", SECRETS)])
    with pytest.raises(NormalizationError) as ei:
        reg.ingest({BANK_ID_HEADER: "nope"}, b"{}")
    assert ei.value.code == "unknown_bank"
    with pytest.raises(NormalizationError) as ei:
        reg.ingest({}, b"{}")
    assert ei.value.code == "unknown_bank"


def test_registry_auth_checked_before_body_parsed():
    reg = AdapterRegistry([FinacleAdapter("sbi", SECRETS)])
    with pytest.raises(AuthenticationError):
        reg.ingest({BANK_ID_HEADER: "sbi", "X-Finacle-Signature": "bad"}, b"not json at all")


def test_registry_bad_json_after_valid_auth():
    reg = AdapterRegistry([FinacleAdapter("sbi", SECRETS)])
    body = b"not json"
    sig = hmac.new(b"sbi-secret", body, hashlib.sha256).hexdigest()
    with pytest.raises(NormalizationError) as ei:
        reg.ingest({BANK_ID_HEADER: "sbi", "X-Finacle-Signature": sig}, body)
    assert ei.value.code == "bad_value" and ei.value.fields == ["<body>"]


def test_registry_json_numbers_become_decimal_not_float():
    reg = AdapterRegistry([TemenosAdapter("hdfc", SECRETS)])
    p = temenos_payload()
    p["body"]["transaction"]["amount"] = 150000.00
    body = json.dumps(p).encode()          # serialises as 150000.0 — a JSON number
    tx = reg.ingest({BANK_ID_HEADER: "hdfc", "X-Client-Cert-Fingerprint": FP}, body)
    assert tx.amount_inr == Decimal("150000.0")


def test_registry_rejects_duplicate_bank():
    reg = AdapterRegistry([FinacleAdapter("sbi", SECRETS)])
    with pytest.raises(ValueError):
        reg.register(TemenosAdapter("SBI", SECRETS))


def test_default_registry_wires_five_banks():
    reg = default_registry(SECRETS)
    assert reg.bank_ids == ["axis", "demo_bancs_bank", "hdfc", "pnb", "sbi"]


def test_env_secret_provider():
    p = EnvSecretProvider({"KAVACH_SECRET__SBI__HMAC_SECRET": "abc"})
    assert p.get("sbi", "hmac_secret") == b"abc"
    assert p.get("pnb", "hmac_secret") is None


# --------------------------------------------------------------- feature contract
def test_feature_contract_shape_is_identical_across_vendors():
    """The whole point of FEATURE_CONTRACT: three different payload shapes -> one vector shape."""
    vecs = [
        FinacleAdapter("sbi", SECRETS).normalize(finacle_payload()).to_feature_vector(),
        TemenosAdapter("hdfc", SECRETS).normalize(temenos_payload()).to_feature_vector(),
        BancsAdapter("demo_bancs_bank", SECRETS).normalize(bancs_payload()).to_feature_vector(),
    ]
    assert all(len(v) == FEATURE_CONTRACT.dim for v in vecs)
    assert FEATURE_CONTRACT.dim == len(FEATURE_CONTRACT.names) == 17


def test_feature_contract_missing_indicator_slots():
    """A bank that never sends an optional field still emits the same-shaped vector, with the indicator lit."""
    full = FinacleAdapter("sbi", SECRETS).normalize(finacle_payload())
    p = finacle_payload()
    del p["FIXML"]["Body"]["CustDtls"]["ACCT_OPN_DAYS"]
    del p["FIXML"]["Body"]["BenefDtls"]["FIRST_TIME_BENEF"]
    sparse = FinacleAdapter("sbi", SECRETS).normalize(p)
    vf, vs = full.to_feature_vector(), sparse.to_feature_vector()
    n = FEATURE_CONTRACT.names
    assert len(vf) == len(vs)
    assert vf[n.index("sender_account_age_days_missing")] == 0.0
    assert vs[n.index("sender_account_age_days_missing")] == 1.0
    assert vs[n.index("log1p_sender_account_age_days")] == FEATURE_CONTRACT.IMPUTE_VALUE
    assert vf[n.index("is_new_beneficiary_missing")] == 0.0
    assert vs[n.index("is_new_beneficiary_missing")] == 1.0
    # tri-state biometric: UNKNOWN -> value imputed, missing=1
    assert vf[n.index("caller_verified_biometric_missing")] == 1.0
    # rail one-hot: exactly one lit
    rails = [vf[i] for i, name in enumerate(n) if name.startswith("rail_")]
    assert sum(rails) == 1.0 and vf[n.index("rail_upi")] == 1.0


def test_feature_contract_excludes_free_text():
    assert not any(f in FEATURE_CONTRACT.names for f in KavachTransaction.FREE_TEXT_FIELDS)
