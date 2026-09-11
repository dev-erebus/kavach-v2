"""Metric plumbing in the federated experiments: lift, win-count invariance, bridge level evaluation, split stats."""
import numpy as np
import pytest

from federated.experiments import bridge_calibration as bc
from federated.experiments.cold_start import N_BANKS, Trial, _agg, _corr_with_prevalence, _wins, make_split


def trial(method, ap, held_out=0, seed=0, n=500, test_n=10_000, test_fraud=20, split="amount"):
    return Trial(split, held_out, seed, n, method, ap, 0.9, test_n, test_fraud, 1)


class TestLift:
    def test_constant_scorer_has_lift_one(self):
        t = trial("local_only", ap=0.002, test_n=10_000, test_fraud=20)  # prevalence 0.002
        assert t.prevalence == pytest.approx(0.002) and t.lift == pytest.approx(1.0)

    def test_lift_scales_with_prevalence_floor(self):
        a = trial("fedavg", ap=0.5, test_fraud=10)  # prev 0.001 -> lift 500
        b = trial("fedavg", ap=0.5, test_fraud=100)  # prev 0.01 -> lift 50
        assert a.lift == pytest.approx(500) and b.lift == pytest.approx(50)

    def test_agg_reports_both_metrics(self):
        ts = [trial("fedavg", 0.5, held_out=h, test_fraud=10 * (h + 1)) for h in range(5)]
        mu_ap, sd_ap, k = _agg(ts, "amount", 500, "fedavg", "auc_pr")
        mu_l, sd_l, _ = _agg(ts, "amount", 500, "fedavg", "lift")
        assert k == 5 and mu_ap == pytest.approx(0.5) and sd_ap == 0.0
        assert sd_l > 0  # same AUC-PR, different floors -> lift differs; that is the point of reporting both


class TestWinsInvariance:
    def test_paired_wins_identical_under_lift(self):
        ts = []
        for h in range(5):
            ts.append(trial("fedavg", 0.6 if h != 2 else 0.1, held_out=h, test_fraud=5 * (h + 1)))
            ts.append(trial("local_only", 0.3, held_out=h, test_fraud=5 * (h + 1)))
        w_ap, tot = _wins(ts, "amount", 500, "fedavg", "local_only")
        # recompute on lift by hand: same cell shares the floor, so ordering is preserved
        lifts_f = {t.held_out: t.lift for t in ts if t.method == "fedavg"}
        lifts_l = {t.held_out: t.lift for t in ts if t.method == "local_only"}
        w_lift = sum(lifts_f[h] > lifts_l[h] for h in range(5))
        assert (w_ap, tot) == (4, 5) and w_lift == w_ap


class TestPrevalenceCorrelation:
    def test_detects_base_rate_driven_spread(self):
        # AUC-PR exactly proportional to prevalence -> r = 1
        ts = [trial("fedavg", ap=0.01 * (h + 1), held_out=h, test_fraud=10 * (h + 1)) for h in range(5)]
        r, k = _corr_with_prevalence(ts, "amount", 500, "fedavg")
        assert k == 5 and r == pytest.approx(1.0)

    def test_nan_when_degenerate(self):
        ts = [trial("fedavg", ap=0.5, held_out=h, test_fraud=10) for h in range(5)]  # identical prevalence
        r, _ = _corr_with_prevalence(ts, "amount", 500, "fedavg")
        assert np.isnan(r)


class TestBridgeEval:
    def test_eval_level_counts(self):
        p = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
        y = np.array([1, 0, 1, 0, 0])
        alerts, tp, fr, prec, rec = bc._eval_level(p, y, cutoff=0.5)
        assert (alerts, tp) == (2, 1) and fr == 0.4 and prec == 0.5 and rec == 0.5

    def test_eval_level_no_alerts_is_nan_precision(self):
        alerts, tp, fr, prec, rec = bc._eval_level(np.array([0.1, 0.2]), np.array([1, 0]), cutoff=0.9)
        assert alerts == 0 and np.isnan(prec) and rec == 0.0


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = 20_000
    X = rng.normal(size=(n, 29))
    amount = rng.lognormal(3, 1.5, size=n)
    y = (rng.uniform(size=n) < 0.01).astype(int)
    return X, amount, y


class TestSplits:
    def test_iid_split_balances_prevalence(self, toy):
        X, amount, y = toy
        s = make_split("iid", X, amount, y, np.random.default_rng(1))
        prevs = [b["prevalence"] for b in s.per_bank]
        assert max(prevs) / min(prevs) < 1.6
        assert sum(b["n"] for b in s.per_bank) == len(y)

    def test_amount_split_orders_banks_by_amount(self, toy):
        X, amount, y = toy
        s = make_split("amount", X, amount, y, np.random.default_rng(1))
        med = [b["median_amount"] for b in s.per_bank]
        assert med == sorted(med)  # bank k skewed to amount band k
        assert set(np.unique(s.bank_of_row)) == set(range(N_BANKS))

    def test_cluster_split_covers_all_banks_and_describes_method(self, toy):
        X, amount, y = toy
        s = make_split("cluster", X, amount, y, np.random.default_rng(1))
        assert all(b["n"] > 0 for b in s.per_bank)
        assert "k-means" in s.method_description and "p=0.7" in s.method_description
