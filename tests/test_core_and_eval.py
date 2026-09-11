"""v1 rule classifier, eval metrics, and the dataset→schema mapping. All offline."""
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from adapters.schema import KavachTransaction, PaymentRail
from core.levels import ThreatLevel
from core.rules import RuleConfig, V1RuleClassifier
from eval import metrics
from eval.dataset_loader import EvalMappingConfig, LabeledDataset, to_kavach_transactions


def tx(**over) -> KavachTransaction:
    base = dict(
        transaction_id="t1", bank_id="sbi", timestamp_utc=datetime(2026, 9, 11, tzinfo=timezone.utc),
        amount_inr="5000", rail=PaymentRail.UPI, sender_age=40, receiver_account_age_days=3650,
        is_active_phone_call=False, device_location="Delhi",
    )
    base.update(over)
    return KavachTransaction(**base)


# ------------------------------------------------------------------ v1 rules
class TestV1Rules:
    clf = V1RuleClassifier()

    def test_default_is_l0(self):
        assert self.clf.classify(tx()).level is ThreatLevel.L0

    def test_l3_needs_amount_call_and_profile(self):
        d = self.clf.classify(tx(amount_inr="150000", is_active_phone_call=True, sender_age=70))
        assert d.level is ThreatLevel.L3 and d.rule == "L3_high_amount_call_and_profile"
        d = self.clf.classify(tx(amount_inr="150000", is_active_phone_call=True, receiver_account_age_days=2))
        assert d.level is ThreatLevel.L3
        # high amount + call but 40-year-old sender paying a 10-year-old account -> L2 not L3
        assert self.clf.classify(tx(amount_inr="150000", is_active_phone_call=True)).level is ThreatLevel.L2

    def test_high_amount_alone_is_l2(self):
        d = self.clf.classify(tx(amount_inr="100000"))
        assert d.level is ThreatLevel.L2 and d.rule == "L2_high_amount_alone"

    def test_call_and_fresh_receiver_is_l2_regardless_of_amount(self):
        d = self.clf.classify(tx(amount_inr="500", is_active_phone_call=True, receiver_account_age_days=1))
        assert d.level is ThreatLevel.L2 and d.rule == "L2_call_and_fresh_receiver"

    def test_biometric_unknown_is_not_a_signal_but_failed_is(self):
        # None (unknown) must not escalate (brief §5.1 / §5.5)
        a = self.clf.classify(tx(amount_inr="500", is_active_phone_call=True, caller_verified_biometric=None))
        assert a.level is ThreatLevel.L1 and a.rule == "L1_call_alone"
        # False (liveness check FAILED -> likely cloned voice) escalates
        b = self.clf.classify(tx(amount_inr="500", is_active_phone_call=True, caller_verified_biometric=False))
        assert b.level is ThreatLevel.L2
        # True (verified human) does not escalate beyond the call itself
        c = self.clf.classify(tx(amount_inr="500", is_active_phone_call=True, caller_verified_biometric=True))
        assert c.level is ThreatLevel.L1

    def test_l1_branches(self):
        assert self.clf.classify(tx(amount_inr="10000")).rule == "L1_low_amount"
        assert self.clf.classify(tx(is_active_phone_call=True)).rule == "L1_call_alone"
        assert self.clf.classify(tx(receiver_account_age_days=20)).rule == "L1_young_receiver"

    def test_config_is_the_only_knob(self):
        strict = V1RuleClassifier(RuleConfig(amount_high_inr=Decimal("5000")))
        assert strict.classify(tx()).level is ThreatLevel.L2

    def test_every_rule_name_is_reachable(self):
        cases = [
            tx(amount_inr="150000", is_active_phone_call=True, sender_age=70),
            tx(amount_inr="100000"),
            tx(is_active_phone_call=True, receiver_account_age_days=1),
            tx(amount_inr="60000", sender_age=65),
            tx(amount_inr="10000"),
            tx(is_active_phone_call=True),
            tx(receiver_account_age_days=20),
            tx(),
        ]
        fired = {self.clf.classify(t).rule for t in cases}
        assert fired == set(V1RuleClassifier.RULE_NAMES)


# ------------------------------------------------------------------- metrics
class TestMetrics:
    def test_points_and_distribution(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
        lv = np.array([0, 1, 2, 3, 3, 2, 0, 1])
        m = metrics.compute(y, lv, rule_names=["a", "b", "c", "d", "d", "c", "a", "b"])
        p2 = m.point(ThreatLevel.L2)
        assert (p2.tp, p2.fp, p2.fn, p2.tn) == (2, 2, 1, 3)
        assert p2.precision == 0.5 and p2.recall == pytest.approx(2 / 3)
        assert p2.flag_rate == 0.5
        p3 = m.point(ThreatLevel.L3)
        assert (p3.tp, p3.fp) == (1, 1)
        assert m.distribution.fraud[ThreatLevel.L3] == 1 and m.distribution.legit[ThreatLevel.L1] == 2
        assert m.rule_attribution["d"] == {"n": 2, "fraud": 1, "legit": 1}
        assert m.score_is_ordinal is True
        assert m.auc_pr_baseline == pytest.approx(3 / 8)

    def test_perfect_ranker_auc_pr_is_one(self):
        y = np.array([0, 0, 1, 1])
        m = metrics.compute(y, np.array([0, 0, 3, 3]))
        assert m.auc_pr == pytest.approx(1.0)

    def test_continuous_score_overrides_ordinal(self):
        y = np.array([0, 1, 0, 1])
        m = metrics.compute(y, np.array([0, 0, 0, 0]), score=np.array([0.1, 0.9, 0.2, 0.8]))
        assert m.score_is_ordinal is False and m.auc_pr == pytest.approx(1.0)

    def test_empty_prediction_bucket_gives_nan_precision_not_crash(self):
        m = metrics.compute(np.array([0, 1]), np.array([0, 0]))
        assert np.isnan(m.point(ThreatLevel.L3).precision) and m.point(ThreatLevel.L3).f1 == 0.0


# ------------------------------------------------------------------- mapping
class TestDatasetMapping:
    def make_ds(self, amounts, labels):
        df = pd.DataFrame({"V1": np.zeros(len(amounts)), "amount": amounts, "label": labels})
        return LabeledDataset("synthetic", "unit test", df, ["V1"])

    def test_rows_go_through_schema_and_fx(self):
        ds = self.make_ds([10.0, 100.5], [0, 1])
        mp = to_kavach_transactions(ds, EvalMappingConfig(fx_to_inr=Decimal("90")))
        assert len(mp.transactions) == 2 and mp.rejected == {} and mp.rejected_fraud == 0
        assert mp.transactions[0].amount_inr == Decimal("900.00")
        assert mp.transactions[1].amount_inr == Decimal("9045.00")
        assert list(mp.labels) == [0, 1]

    def test_neutral_fills_cannot_fire_any_social_engineering_rule(self):
        ds = self.make_ds([1.0], [0])
        t = to_kavach_transactions(ds).transactions[0]
        assert t.is_active_phone_call is False
        assert t.caller_verified_biometric is None
        assert t.sender_age < RuleConfig().elderly_age
        assert t.receiver_account_age_days > RuleConfig().young_receiver_days
        assert V1RuleClassifier().classify(t).rule == "L0_default"

    def test_zero_amount_rows_rejected_and_fraud_among_them_counted(self):
        ds = self.make_ds([0.0, 5.0, 0.0], [1, 0, 0])
        mp = to_kavach_transactions(ds)
        assert len(mp.transactions) == 1
        assert sum(mp.rejected.values()) == 2 and mp.rejected_fraud == 1
        assert list(mp.kept_index) == [1]
