"""Component C (FR-2 template generator) and §5.1 (voice-liveness interface)."""
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from adapters.schema import KavachTransaction, PaymentRail
from compliance import fr2_generator as fr2
from compliance.field_mapping import FIELDS, Source, unverified
from core.levels import ThreatLevel
from hardening.voice_liveness import (
    AudioSample,
    LivenessVerdict,
    NullLivenessProvider,
    StubLivenessProvider,
    VoiceLivenessProvider,
    resolve_liveness,
)

T0 = datetime(2026, 9, 11, 8, 33, 22, tzinfo=timezone.utc)


def tx(**over) -> KavachTransaction:
    d = dict(
        transaction_id="SBI-1", bank_id="sbi", timestamp_utc=T0, amount_inr=Decimal("150000"), rail=PaymentRail.UPI,
        sender_age=67, receiver_account_age_days=3, is_active_phone_call=True, caller_verified_biometric=False,
        is_new_beneficiary=True, device_location="Lucknow, UP", receiver_name="R KUMAR", remarks="urgent",
    )
    d.update(over)
    return KavachTransaction(**d)


def meta(**over) -> fr2.IncidentMeta:
    d = dict(incident_id="KV-2026-000123", detected_at_utc=T0 + timedelta(seconds=4), level=ThreatLevel.L3,
             decided_by="v1_rules:L3_high_amount_call_and_profile", funds_moved=False,
             tampering_flags={"remarks": ["override_phrase"]}, dossier_reference="dossier://KV-2026-000123")
    d.update(over)
    return fr2.IncidentMeta(**d)


class TestFR2Template:
    def test_source_files_carry_the_compliance_banner(self):
        for name in ("fr2_generator.py", "field_mapping.py"):
            first = open(f"compliance/{name}", encoding="utf-8").readline()
            assert first.strip() == "# COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE"

    def test_never_filing_ready_and_assert_raises(self):
        r = fr2.generate(tx(), meta(), now=T0 + timedelta(hours=6))
        assert r.filing_ready is False
        assert set(r.unverified_fields) == {f.key for f in FIELDS}  # nothing verified yet
        with pytest.raises(RuntimeError, match="TEMPLATE"):
            r.assert_filing_ready()

    def test_manual_fields_are_placeholders_not_guesses(self):
        r = fr2.generate(tx(), meta(), now=T0 + timedelta(hours=6))
        manual_keys = {f.key for f in FIELDS if f.source is Source.MANUAL}
        assert set(r.manual_fields) == manual_keys
        for k in manual_keys:
            assert str(r.values[k]).startswith("<<MANUAL:")
        # the regulatory classification is never auto-filled
        assert r.values["fraud_category"].startswith("<<MANUAL:")

    def test_auto_fields_derived_and_data_minimised(self):
        r = fr2.generate(tx(), meta(), now=T0 + timedelta(hours=6))
        v = r.values
        assert v["amount_involved_inr"] == "150000"
        assert v["bank_transaction_reference"] == "SBI-1"
        assert v["date_of_occurrence"] == "2026-09-11T14:03:22+05:30"  # UTC -> IST
        assert v["detection_to_report_hours"] == pytest.approx(6.0, abs=0.01)
        assert v["victim_customer_age_band"] == "60–74 years"  # band, not 67
        assert "67" not in json.dumps(v["victim_customer_age_band"])
        assert v["beneficiary_account_vintage"] == "0–6 days"
        assert v["is_attempted_or_completed"].startswith("attempted")
        assert "voice-liveness check FAILED" in v["suspected_modus_operandi"]
        assert "prompt-injection" in v["suspected_modus_operandi"]
        assert v["social_engineering_indicators"]["voice_liveness"] == "FAILED"
        assert v["social_engineering_indicators"]["free_text_tampering_flags"] == {"remarks": ["override_phrase"]}

    def test_unknown_biometric_reported_as_unknown_not_false(self):
        r = fr2.generate(tx(caller_verified_biometric=None), meta(funds_moved=None), now=T0 + timedelta(hours=1))
        assert r.values["social_engineering_indicators"]["voice_liveness"] == "unknown/not available"
        assert r.values["is_attempted_or_completed"] == "unknown at report time"
        assert "FAILED" not in r.values["suspected_modus_operandi"]

    def test_only_l3_eligible(self):
        with pytest.raises(ValueError):
            fr2.generate(tx(), meta(level=ThreatLevel.L2))

    def test_outputs_carry_the_notice(self):
        r = fr2.generate(tx(), meta(), now=T0 + timedelta(hours=6))
        j = json.loads(r.to_json())
        assert "COMPLIANCE REVIEW REQUIRED" in j["_notice"] and j["filing_ready"] is False
        md = r.to_markdown()
        assert "COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE" in md
        assert "- [ ] `fraud_category`" in md  # outstanding checklist
        assert "Kavach classifies risk" in md


class TestVoiceLiveness:
    sample = AudioSample(reference="call-77", duration_ms=3000)

    def test_null_provider_is_unknown(self):
        assert resolve_liveness(NullLivenessProvider(), self.sample, True) is None

    def test_no_call_no_provider_no_sample_all_unknown(self):
        p = StubLivenessProvider(LivenessVerdict.SYNTHETIC)
        assert resolve_liveness(p, self.sample, is_active_phone_call=False) is None
        assert resolve_liveness(None, self.sample, True) is None
        assert resolve_liveness(p, None, True) is None

    def test_confident_verdicts_map_to_tristate(self):
        assert resolve_liveness(StubLivenessProvider(LivenessVerdict.LIVE_HUMAN), self.sample, True) is True
        assert resolve_liveness(StubLivenessProvider(LivenessVerdict.SYNTHETIC), self.sample, True) is False

    def test_low_confidence_is_unknown_not_a_guess(self):
        assert resolve_liveness(StubLivenessProvider(LivenessVerdict.SYNTHETIC, confidence=0.6), self.sample, True) is None

    def test_provider_failure_and_exception_degrade_to_unknown(self):
        assert resolve_liveness(StubLivenessProvider(LivenessVerdict.SYNTHETIC, fail=True), self.sample, True) is None

        class Broken(VoiceLivenessProvider):
            name = "broken"

            def check(self, sample, timeout_ms=800):
                raise TimeoutError("provider hung")

        assert resolve_liveness(Broken(), self.sample, True) is None

    def test_result_flows_into_schema_without_coercion(self):
        v = resolve_liveness(NullLivenessProvider(), self.sample, True)
        t = tx(caller_verified_biometric=v)
        assert t.caller_verified_biometric is None
        from adapters.schema import FEATURE_CONTRACT
        vec = t.to_feature_vector()
        assert vec[FEATURE_CONTRACT.names.index("caller_verified_biometric_missing")] == 1.0
        assert vec[FEATURE_CONTRACT.names.index("caller_verified_biometric")] == FEATURE_CONTRACT.IMPUTE_VALUE
