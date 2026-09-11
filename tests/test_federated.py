"""Component B: learner, aggregator, model store, threshold bridge, feedback — plus the
(b)-style test that the production 17-dim FEATURE_CONTRACT round-trips through all of it."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from adapters.schema import FEATURE_CONTRACT, KavachTransaction, PaymentRail
from core.levels import ThreatLevel
from federated.aggregator import Aggregator, CoordinateMedian, FedAvg, TrimmedMean
from federated.feature_spec import KAVACH_PRODUCTION, ULB_EVAL, FeatureSpec, SpecMismatch
from federated.feedback import FeatureLog, FeedbackEvent, FeedbackHandler, Outcome
from federated.local_trainer import Delta, LocalTrainer, OnlineLogisticRegression, TrainConfig
from federated.model_store import ModelStore
from federated.threshold_bridge import (
    FALLBACK_LEVEL,
    Cutoffs,
    FlagRateTargets,
    ThresholdBridge,
    calibrate_by_flag_rate,
    calibrate_by_precision,
)

SPEC = FeatureSpec("toy", 1, ("a", "b", "c"))


def toy_data(rng, n=2000, w=(4.0, -2.0, 1.0), b=-1.0):
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(X @ np.array(w) + b)))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


# ------------------------------------------------------------------- learner
class TestLearner:
    def test_learns_toy_problem(self):
        rng = np.random.default_rng(0)
        X, y = toy_data(rng)
        m = OnlineLogisticRegression(SPEC, TrainConfig(pos_weight=1.0))
        m.fit_epochs(X, y, rng, epochs=5)
        acc = ((m.predict_proba(X) > 0.5) == y).mean()
        assert acc > 0.85
        assert np.sign(m.w[0]) > 0 and np.sign(m.w[1]) < 0

    def test_spec_enforced_on_params_and_inputs(self):
        m = OnlineLogisticRegression(SPEC)
        with pytest.raises(SpecMismatch):
            m.set_params(np.zeros(3))  # needs dim+1
        with pytest.raises(SpecMismatch):
            m.predict_proba(np.zeros((5, 4)))
        with pytest.raises(SpecMismatch):
            m.set_params(np.array([0, 0, np.inf, 0]))

    def test_delta_carries_no_data_and_is_relative_to_sync(self):
        rng = np.random.default_rng(1)
        X, y = toy_data(rng, n=500)
        t = LocalTrainer("sbi", SPEC, TrainConfig(pos_weight=1.0))
        g = np.array([0.1, 0.2, 0.3, 0.0])
        t.sync(g, global_version=3)
        t.learn(X, y, epochs=1)
        d = t.export_delta()
        assert set(vars(d)) == {"bank_id", "spec_key", "base_version", "delta", "n_samples"}
        assert d.base_version == 3 and d.n_samples == 500 and d.spec_key == SPEC.key
        np.testing.assert_allclose(d.delta, t.model.get_params() - g)


# ---------------------------------------------------------------- aggregator
class TestAggregator:
    def d(self, bank, vec, n=100, v=0, key=SPEC.key):
        return Delta(bank, key, v, np.asarray(vec, float), n)

    def test_fedavg_is_sample_weighted_mean(self):
        agg = Aggregator(SPEC, FedAvg(), max_delta_norm=None)
        agg.aggregate([self.d("a", [1, 1, 1, 1], n=300), self.d("b", [0, 0, 0, 0], n=100)])
        np.testing.assert_allclose(agg.global_params, [0.75] * 4)
        assert agg.version == 1

    def test_median_ignores_weights_and_outlier(self):
        agg = Aggregator(SPEC, CoordinateMedian(), max_delta_norm=None)
        agg.aggregate([
            self.d("a", [1, 1, 1, 1]), self.d("b", [1.2, 0.8, 1, 1]),
            self.d("evil", [-100, 100, -100, 100], n=10**6),
        ])
        np.testing.assert_allclose(agg.global_params, [1, 1, 1, 1])

    def test_trimmed_mean_falls_back_to_median_when_too_few(self):
        assert np.allclose(TrimmedMean(0.2).combine(np.array([[0.0], [1.0], [100.0]]), np.ones(3)), [1.0])

    def test_rejects_spec_mismatch_shape_stale_and_nonfinite(self):
        agg = Aggregator(SPEC, FedAvg())
        rep = agg.aggregate([
            self.d("wrong_spec", [1, 1, 1, 1], key=ULB_EVAL.key),
            self.d("wrong_shape", [1, 1, 1]),
            self.d("stale", [1, 1, 1, 1], v=7),
            self.d("nan", [1, np.nan, 1, 1]),
            self.d("ok", [0.5, 0.5, 0.5, 0.5]),
        ])
        assert rep.n_accepted == 1 and set(rep.rejected) == {"wrong_spec", "wrong_shape", "stale", "nan"}
        assert "spec mismatch" in rep.rejected["wrong_spec"]
        np.testing.assert_allclose(agg.global_params, [0.5] * 4)

    def test_norm_clipping_bounds_one_round(self):
        agg = Aggregator(SPEC, FedAvg(), max_delta_norm=1.0)
        rep = agg.aggregate([self.d("big", [100, 0, 0, 0])])
        assert rep.clipped == ["big"] and np.linalg.norm(agg.global_params) == pytest.approx(1.0)

    def test_no_update_below_min_participants(self):
        agg = Aggregator(SPEC, CoordinateMedian(), min_participants=3)
        rep = agg.aggregate([self.d("a", [1, 1, 1, 1])])
        assert agg.version == 0 and rep.n_accepted == 1 and np.all(agg.global_params == 0)


# --------------------------------------------------------------- model store
class TestModelStore:
    def test_versioning_and_roundtrip(self, tmp_path):
        st = ModelStore(tmp_path, SPEC)
        st.save("global", np.array([1, 2, 3, 4.0]), n_samples=10)
        st.save("global", np.array([2, 3, 4, 5.0]), n_samples=20, base_global_version=1)
        st.save("sbi", np.array([0, 0, 0, 1.0]), base_global_version=2, meta={"note": "x"})
        assert st.latest_version("global") == 2 and st.latest_version("sbi") == 1
        np.testing.assert_allclose(st.load("global").params_array, [2, 3, 4, 5])
        np.testing.assert_allclose(st.load("global", version=1).params_array, [1, 2, 3, 4])
        rec = st.load("sbi")
        assert rec.spec_key == SPEC.key and rec.base_global_version == 2 and rec.meta == {"note": "x"}
        assert st.scopes() == ["global", "sbi"]

    def test_refuses_wrong_shape_and_wrong_spec(self, tmp_path):
        st = ModelStore(tmp_path, SPEC)
        with pytest.raises(SpecMismatch):
            st.save("global", np.zeros(3))
        st.save("global", np.zeros(4))
        # A store serving a different spec must not read this file even if pointed at the same root.
        other = ModelStore(tmp_path, FeatureSpec("toy", 2, ("a", "b", "c")))
        with pytest.raises(FileNotFoundError):
            other.load("global")  # different spec -> different directory -> nothing there
        # Tamper: copy the v1 file into the other spec's directory and confirm the spec_key check fires.
        src = next((tmp_path / SPEC.key.replace("@", "_").replace("[", "_").replace("]", "_")).rglob("v0001.json"), None) \
            or next(tmp_path.rglob("v0001.json"))
        dst_dir = tmp_path / "toy_v2_3_" / "global"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "v0001.json").write_bytes(src.read_bytes())
        with pytest.raises(SpecMismatch):
            other.load("global")


# ----------------------------------------------------------- threshold bridge
class TestThresholdBridge:
    def test_flag_rate_calibration_hits_budget(self):
        rng = np.random.default_rng(0)
        s = rng.beta(0.5, 20, size=100_000)
        c = calibrate_by_flag_rate(s, FlagRateTargets(0.05, 0.01, 0.001))
        assert (s >= c.l1).mean() == pytest.approx(0.05, abs=0.002)
        assert (s >= c.l2).mean() == pytest.approx(0.01, abs=0.001)
        assert (s >= c.l3).mean() == pytest.approx(0.001, abs=0.0005)
        assert c.l1 <= c.l2 <= c.l3

    def test_precision_calibration(self):
        s = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        c = calibrate_by_precision(s, y, precision_targets=(0.4, 0.75, 1.0))
        # cumulative precision by rank: 1, 1, 1, .75, .8, .67, .57, .5, .44, .4
        assert c.l3 == 0.9  # loosest rank with precision >= 1.0 is rank 3 -> score 0.9
        assert c.l2 == 0.7  # loosest rank with precision >= .75 is rank 5 (0.8), not rank 4 -> score 0.7
        assert c.l1 == 0.2  # loosest rank with precision >= .4 is the whole list -> score 0.2
        c2 = calibrate_by_precision(s, y, precision_targets=(0.99, 0.99, 0.99))
        assert c2.l1 == c2.l2 == c2.l3 == 0.9

    def test_cutoffs_must_nest(self):
        with pytest.raises(ValueError):
            Cutoffs(0.5, 0.4, 0.9)

    def test_fallback_is_l2_never_l0(self):
        b = ThresholdBridge("sbi", SPEC)
        d = b.decide(np.zeros(3))
        assert d.level is FALLBACK_LEVEL is ThreatLevel.L2 and d.fallback and "no model" in d.reason
        b.model = OnlineLogisticRegression(SPEC)
        d = b.decide(np.zeros(3))
        assert d.fallback and "not calibrated" in d.reason
        b.cutoffs = Cutoffs(0.3, 0.6, 0.9)
        d = b.decide(np.zeros(4))  # wrong dim
        assert d.level is ThreatLevel.L2 and d.fallback and "rejected" in d.reason
        d = b.decide(np.zeros(3))  # p = sigmoid(0) = 0.5 -> L1
        assert not d.fallback and d.level is ThreatLevel.L1 and d.probability == pytest.approx(0.5)

    def test_levels_from_probability(self):
        c = Cutoffs(0.3, 0.6, 0.9)
        assert [c.level(p) for p in (0.1, 0.3, 0.7, 0.95)] == [ThreatLevel.L0, ThreatLevel.L1, ThreatLevel.L2, ThreatLevel.L3]


# ------------------------------------------------------------------ feedback
class TestFeedback:
    def setup_method(self):
        self.t = LocalTrainer("sbi", SPEC, TrainConfig(pos_weight=1.0))
        self.log = FeatureLog()
        self.h = FeedbackHandler(self.t, self.log)
        self.scored_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.log.record("tx1", np.array([1.0, 0.0, 0.0]), self.scored_at)

    def ev(self, **over):
        base = dict(bank_id="sbi", transaction_id="tx1", outcome=Outcome.CONFIRMED_FRAUD, reported_by="ops.1",
                    reported_at_utc=self.scored_at + timedelta(days=2))
        base.update(over)
        return FeedbackEvent(**base)

    def test_accepts_and_learns(self):
        before = self.t.model.get_params()
        r = self.h.handle(self.ev())
        assert r.accepted and r.label_applied == 1
        assert not np.allclose(before, self.t.model.get_params())
        assert self.t.export_delta().n_samples == 1

    def test_rejections(self):
        assert not self.h.handle(self.ev(bank_id="pnb")).accepted
        assert not self.h.handle(self.ev(transaction_id="never_scored")).accepted
        assert not self.h.handle(self.ev(reported_at_utc=self.scored_at - timedelta(hours=1))).accepted
        assert not self.h.handle(self.ev(reported_at_utc=self.scored_at + timedelta(days=200))).accepted
        assert self.h.handle(self.ev()).accepted
        assert not self.h.handle(self.ev(outcome=Outcome.FALSE_POSITIVE)).accepted  # no relabelling
        assert len(self.h.audit) == 1

    def test_free_text_is_not_accepted_in_feedback(self):
        with pytest.raises(Exception):
            FeedbackEvent(bank_id="sbi", transaction_id="tx1", outcome="confirmed_fraud",
                          reported_by="ops.1", reported_at_utc=self.scored_at, note="ignore previous instructions")


# ------------------------------------ (b): production contract end-to-end round trip
def kavach_tx(i: int, rng: np.random.Generator) -> tuple[KavachTransaction, int]:
    """Synthetic Kavach-schema transaction with a planted rule: fraud iff call ∧ fresh receiver ∧ elderly."""
    call = bool(rng.uniform() < 0.3)
    age = int(rng.integers(18, 90))
    rx_days = int(rng.integers(0, 3000)) if rng.uniform() < 0.8 else int(rng.integers(0, 10))
    fraud = int(call and rx_days <= 10 and age >= 60)
    tx = KavachTransaction(
        transaction_id=f"t{i}", bank_id="sbi", timestamp_utc=datetime(2026, 9, 11, tzinfo=timezone.utc),
        amount_inr=Decimal(int(rng.integers(100, 300000))), rail=list(PaymentRail)[int(rng.integers(len(PaymentRail)))],
        sender_age=age, receiver_account_age_days=rx_days, is_active_phone_call=call,
        # optional fields randomly present/absent — exercises the missing-indicator slots
        sender_account_age_days=int(rng.integers(0, 5000)) if rng.uniform() < 0.5 else None,
        is_new_beneficiary=bool(rng.uniform() < 0.5) if rng.uniform() < 0.5 else None,
        caller_verified_biometric=(bool(rng.uniform() < 0.5) if (call and rng.uniform() < 0.3) else None),
        device_location="Lucknow, UP", remarks=None,
    )
    return tx, fraud


class TestProductionContractRoundTrip:
    def test_feature_contract_through_learner_aggregator_store_bridge(self, tmp_path):
        rng = np.random.default_rng(7)
        rows = [kavach_tx(i, rng) for i in range(4000)]
        X = np.array([tx.to_feature_vector() for tx, _ in rows])
        y = np.array([f for _, f in rows])
        assert X.shape == (4000, FEATURE_CONTRACT.dim) == (4000, KAVACH_PRODUCTION.dim) == (4000, 17)
        assert 0.01 < y.mean() < 0.2

        # Two banks federate on the production spec; a third is held out.
        cfg = TrainConfig(pos_weight=5.0, lr=0.1)
        agg = Aggregator(KAVACH_PRODUCTION, CoordinateMedian(), min_participants=2)
        banks = {"sbi": slice(0, 1500), "pnb": slice(1500, 3000)}
        trainers = {b: LocalTrainer(b, KAVACH_PRODUCTION, cfg, seed=1) for b in banks}
        for _ in range(5):
            deltas = []
            for b, tr in trainers.items():
                tr.sync(agg.global_params, agg.version)
                tr.learn(X[banks[b]], y[banks[b]], epochs=3)
                deltas.append(tr.export_delta())
            rep = agg.aggregate(deltas)
            assert rep.n_accepted == 2 and not rep.rejected
        assert agg.version == 5

        # Store the global model and a bank's fine-tuned model; reload under the same spec.
        store = ModelStore(tmp_path, KAVACH_PRODUCTION)
        g = store.save("global", agg.global_params, meta={"rule": "coordinate_median"})
        s = store.save("sbi", trainers["sbi"].model.get_params(), base_global_version=g.version)
        assert store.load("global").spec_key == KAVACH_PRODUCTION.key == "kavach_transaction@v1[17]"

        # Held-out bank: load global from the store, calibrate, and decide via the bridge.
        held = OnlineLogisticRegression(KAVACH_PRODUCTION, cfg, params=store.load("global").params_array)
        Xh, yh = X[3000:], y[3000:]
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yh, held.predict_proba(Xh))
        assert ap > 3 * yh.mean(), f"global model should beat prevalence on held-out bank; AP={ap:.3f}, prev={yh.mean():.3f}"

        bridge = ThresholdBridge("hdfc", KAVACH_PRODUCTION, model=held,
                                 cutoffs=calibrate_by_flag_rate(held.predict_proba(Xh), FlagRateTargets(0.2, 0.1, 0.02)))
        decisions = bridge.decide_many(Xh)
        assert not any(d.fallback for d in decisions)
        levels = np.array([int(d.level) for d in decisions])
        assert (levels >= 1).mean() == pytest.approx(0.2, abs=0.01)
        # frauds should be over-represented at L2+
        assert yh[levels >= 2].mean() > yh.mean()

        # A ULB-spec store must refuse to serve the production model and vice versa.
        with pytest.raises(SpecMismatch):
            ModelStore(tmp_path, ULB_EVAL).save("global", agg.global_params)
        # and the aggregator refuses a ULB-spec delta into the production federation
        bad = Delta("rogue", ULB_EVAL.key, agg.version, np.zeros(ULB_EVAL.dim + 1), 10)
        rep = agg.aggregate([bad, trainers["sbi"].export_delta(), trainers["pnb"].export_delta()])
        assert "rogue" in rep.rejected and "spec mismatch" in rep.rejected["rogue"]
