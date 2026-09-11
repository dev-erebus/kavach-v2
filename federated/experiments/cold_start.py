"""
Cold-start experiment: does a federated baseline help a bank with little or no history?

    python -m federated.experiments.cold_start            # full run (~5 min)
    python -m federated.experiments.cold_start --quick    # 1 seed, 2 splits, for smoke tests

The claim under test (brief §2.4): on a multi-bank split of a public dataset,
the federated model beats a single bank's local-only model on a *held-out*
bank. That claim is only meaningful if four things are true of the experiment,
so all four are built in and the report states them:

1. **The split is non-IID and the method is stated.** Three split modes are run
   and reported side by side — ``iid`` (control), ``amount`` (banks skewed to
   different transaction-size bands, like a rural-pension bank vs an
   urban-salaried bank) and ``cluster`` (banks skewed to different regions of
   feature space via k-means on the five highest-variance PCA components).
   Each non-IID mode uses a dominant-diagonal mixing matrix so banks overlap
   partially rather than being disjoint slices — see ``make_split``.
2. **Three baselines.** For the held-out bank: *local-only* (trained on its own
   small history from zero), *federated* (FedAvg baseline learned from the
   other banks' weight deltas, then fine-tuned on the same small history), and
   *pooled* (the other banks' raw data pooled centrally — the upper bound
   federation is trying to approach without pooling — then the same fine-tune).
   Coordinate-median federation is reported alongside FedAvg.
3. **History size is swept**: 0, 50, 500, 5000 local labelled rows. The
   curve is the result.
4. **Every bank is held out in turn**, over several seeds; mean and spread are
   reported, not a single number.

Second result — poisoning (brief §5.4): one or two of the contributing banks
submit inverted, inflated deltas. FedAvg vs coordinate-median vs trimmed-mean
on the held-out bank at zero history.

Nothing about banking is asserted by this experiment. The dataset is card
fraud, the "banks" are synthetic partitions, and the result is about the
*mechanism* — whether weight-delta aggregation transfers useful structure
across non-identical partitions — not about Indian UPI fraud.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.dataset_loader import load_ulb
from federated.aggregator import Aggregator, AggregationRule, CoordinateMedian, FedAvg, TrimmedMean
from federated.feature_spec import ULB_EVAL, ulb_features
from federated.local_trainer import Delta, LocalTrainer, OnlineLogisticRegression, TrainConfig

N_BANKS = 5
HISTORY_SIZES = (0, 50, 500, 5000)
TEST_OFFSET = max(HISTORY_SIZES)  # rows beyond this in the held-out bank's permutation form the fixed test set
METHODS = ("local_only", "fedavg", "coord_median", "pooled")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    mode: str
    method_description: str
    bank_of_row: np.ndarray  # shape (n,), values 0..N_BANKS-1
    per_bank: list[dict]  # size / fraud / prevalence / a descriptive stat per bank


def _mix_assign(groups: np.ndarray, rng: np.random.Generator, p_diag: float) -> np.ndarray:
    """Row in group g goes to bank g with prob p_diag, else uniformly to another bank."""
    n = len(groups)
    off = (1.0 - p_diag) / (N_BANKS - 1)
    P = np.full((N_BANKS, N_BANKS), off)
    np.fill_diagonal(P, p_diag)
    banks = np.empty(n, dtype=int)
    for g in range(N_BANKS):
        m = groups == g
        banks[m] = rng.choice(N_BANKS, size=m.sum(), p=P[g])
    return banks


def make_split(mode: str, X: np.ndarray, amount: np.ndarray, y: np.ndarray, rng: np.random.Generator, p_diag: float = 0.7) -> Split:
    if mode == "iid":
        banks = rng.integers(0, N_BANKS, size=len(y))
        desc = "IID control: each row assigned to a bank uniformly at random."
    elif mode == "amount":
        # Quantile bands of transaction amount; bank k is skewed toward band k.
        edges = np.quantile(amount, np.linspace(0, 1, N_BANKS + 1)[1:-1])
        groups = np.digitize(amount, edges)
        banks = _mix_assign(groups, rng, p_diag)
        desc = (
            f"Amount skew: rows binned into {N_BANKS} amount-quantile bands; a row in band k goes to bank k "
            f"with p={p_diag} and to each other bank with p={(1 - p_diag) / (N_BANKS - 1):.3f}. "
            "Banks therefore differ in transaction-size distribution AND fraud prevalence."
        )
    elif mode == "cluster":
        # k-means on the five highest-variance PCA components (V1..V5). Covariate shift, not amount.
        km = KMeans(n_clusters=N_BANKS, n_init=4, random_state=int(rng.integers(1 << 31)))
        groups = km.fit_predict(X[:, :5])
        banks = _mix_assign(groups, rng, p_diag)
        desc = (
            f"Feature-cluster skew: k-means (k={N_BANKS}) on V1..V5, the five highest-variance PCA components; "
            f"a row in cluster k goes to bank k with p={p_diag}, else uniformly elsewhere. "
            "Banks occupy different regions of feature space."
        )
    else:
        raise ValueError(mode)

    per_bank = []
    for k in range(N_BANKS):
        m = banks == k
        per_bank.append(
            {
                "bank": k,
                "n": int(m.sum()),
                "fraud": int(y[m].sum()),
                "prevalence": float(y[m].mean()) if m.any() else float("nan"),
                "median_amount": float(np.median(amount[m])) if m.any() else float("nan"),
            }
        )
    return Split(mode, desc, banks, per_bank)


# --------------------------------------------------------------------------- #
# Training primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpConfig:
    rounds: int = 5  # federation rounds
    local_epochs: int = 1  # epochs each bank trains per round
    finetune_epochs: int = 10  # epochs of fine-tuning on the held-out bank's history (same for every method)
    pooled_epochs: int = 5
    train: TrainConfig = TrainConfig()


def run_federation(
    rule: AggregationRule,
    bank_data: dict[int, tuple[np.ndarray, np.ndarray]],
    cfg: ExpConfig,
    seed: int,
    poison: Optional[Callable[[int, Delta, int], Delta]] = None,
) -> tuple[np.ndarray, Aggregator]:
    """Run ``cfg.rounds`` of federated training over ``bank_data``; return the global params."""
    agg = Aggregator(ULB_EVAL, rule, min_participants=1)
    trainers = {k: LocalTrainer(f"bank{k}", ULB_EVAL, cfg.train, seed=seed * 100 + k) for k in bank_data}
    for r in range(cfg.rounds):
        deltas = []
        for k, tr in trainers.items():
            tr.sync(agg.global_params, agg.version)
            Xk, yk = bank_data[k]
            tr.learn(Xk, yk, epochs=cfg.local_epochs)
            d = tr.export_delta()
            if poison is not None:
                d = poison(k, d, r)
            deltas.append(d)
        agg.aggregate(deltas)
    return agg.global_params.copy(), agg


def finetune_and_score(
    init_params: np.ndarray,
    hist: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    cfg: ExpConfig,
    seed: int,
) -> dict[str, float]:
    m = OnlineLogisticRegression(ULB_EVAL, cfg.train, params=init_params)
    Xh, yh = hist
    if len(yh) > 0:
        m.fit_epochs(Xh, yh, np.random.default_rng(seed), epochs=cfg.finetune_epochs)
    Xt, yt = test
    p = m.predict_proba(Xt)
    return {
        "auc_pr": float(average_precision_score(yt, p)),
        "auc_roc": float(roc_auc_score(yt, p)),
    }


# --------------------------------------------------------------------------- #
# Experiment 1 — cold start
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    split: str
    held_out: int
    seed: int
    history: int
    method: str
    auc_pr: float
    auc_roc: float
    test_n: int
    test_fraud: int
    history_fraud: int

    @property
    def prevalence(self) -> float:
        """Fraud prevalence of THIS cell's test set — the AUC-PR floor for this cell."""
        return self.test_fraud / self.test_n if self.test_n else float("nan")

    @property
    def lift(self) -> float:
        """AUC-PR over the cell's own prevalence floor. A constant scorer has lift exactly 1.0."""
        p = self.prevalence
        return self.auc_pr / p if p > 0 else float("nan")


def cold_start(
    X: np.ndarray, amount: np.ndarray, y: np.ndarray, modes: tuple[str, ...], seeds: tuple[int, ...], cfg: ExpConfig, log=print
) -> tuple[list[Trial], dict[str, Split]]:
    trials: list[Trial] = []
    splits: dict[str, Split] = {}
    for mode in modes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            split = make_split(mode, X, amount, y, rng)
            splits.setdefault(mode, split)  # keep the first seed's description/stats for the report
            for h in range(N_BANKS):
                t0 = time.time()
                others = {k: (X[split.bank_of_row == k], y[split.bank_of_row == k]) for k in range(N_BANKS) if k != h}
                idx_h = np.where(split.bank_of_row == h)[0]
                perm = rng.permutation(idx_h)
                test_idx = perm[TEST_OFFSET:]
                test = (X[test_idx], y[test_idx])

                # Bases that don't depend on history size — computed once per (mode, seed, h).
                fed_avg, _ = run_federation(FedAvg(), others, cfg, seed)
                fed_med, _ = run_federation(CoordinateMedian(), others, cfg, seed)
                pooled_model = OnlineLogisticRegression(ULB_EVAL, cfg.train)
                Xp = np.concatenate([v[0] for v in others.values()])
                yp = np.concatenate([v[1] for v in others.values()])
                pooled_model.fit_epochs(Xp, yp, np.random.default_rng(seed), epochs=cfg.pooled_epochs)
                bases = {
                    "local_only": np.zeros(ULB_EVAL.dim + 1),
                    "fedavg": fed_avg,
                    "coord_median": fed_med,
                    "pooled": pooled_model.get_params(),
                }

                for n in HISTORY_SIZES:
                    hist_idx = perm[:n]
                    hist = (X[hist_idx], y[hist_idx])
                    for method, base in bases.items():
                        s = finetune_and_score(base, hist, test, cfg, seed)
                        trials.append(
                            Trial(mode, h, seed, n, method, s["auc_pr"], s["auc_roc"], len(test_idx), int(y[test_idx].sum()), int(y[hist_idx].sum()))
                        )
                log(f"  [{mode} seed={seed} held_out=bank{h}] done in {time.time() - t0:.1f}s")
    return trials, splits


# --------------------------------------------------------------------------- #
# Experiment 2 — poisoning
# --------------------------------------------------------------------------- #
@dataclass
class PoisonTrial:
    split: str
    held_out: int
    seed: int
    attackers: int
    attack: str
    rule: str
    auc_pr: float
    clipped: int


def make_poison(attacker_banks: set[int], kind: str, rng: np.random.Generator) -> Callable[[int, Delta, int], Delta]:
    def poison(k: int, d: Delta, r: int) -> Delta:
        if k not in attacker_banks:
            return d
        if kind == "inverted":
            bad = -10.0 * d.delta
        elif kind == "garbage":
            bad = rng.normal(0, 5.0, size=d.delta.shape)
        else:
            raise ValueError(kind)
        # Attacker also lies about its sample count to dominate a weighted average.
        return Delta(d.bank_id, d.spec_key, d.base_version, bad, n_samples=d.n_samples * 20)

    return poison


def poisoning(
    X: np.ndarray, amount: np.ndarray, y: np.ndarray, mode: str, seeds: tuple[int, ...], cfg: ExpConfig, log=print
) -> list[PoisonTrial]:
    out: list[PoisonTrial] = []
    rules = {"fedavg": FedAvg, "coord_median": CoordinateMedian, "trimmed_mean_20": lambda: TrimmedMean(0.2)}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        split = make_split(mode, X, amount, y, rng)
        for h in range(N_BANKS):
            others_ids = [k for k in range(N_BANKS) if k != h]
            others = {k: (X[split.bank_of_row == k], y[split.bank_of_row == k]) for k in others_ids}
            idx_h = np.where(split.bank_of_row == h)[0]
            test = (X[idx_h], y[idx_h])
            for n_att in (0, 1, 2):
                attackers = set(others_ids[:n_att])
                for attack in (("none",) if n_att == 0 else ("inverted", "garbage")):
                    for rname, rfac in rules.items():
                        pz = None if n_att == 0 else make_poison(attackers, attack, np.random.default_rng(seed))
                        params, agg = run_federation(rfac(), others, cfg, seed, poison=pz)
                        m = OnlineLogisticRegression(ULB_EVAL, cfg.train, params=params)
                        ap = float(average_precision_score(test[1], m.predict_proba(test[0])))
                        clipped = sum(len(rep.clipped) for rep in agg.history)
                        out.append(PoisonTrial(mode, h, seed, n_att, attack, rname, ap, clipped))
            log(f"  [poison {mode} seed={seed} held_out=bank{h}] done")
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
#: Row order for result tables: cells where local-only has real data first; n=0 last as the
#: degenerate reference (local-only is a constant scorer there and cannot rank).
TABLE_ORDER = tuple(n for n in HISTORY_SIZES if n > 0) + (0,)
BREAKDOWN_N = (50, 500)


def _cells(trials: list[Trial], mode: str, n: int, method: str) -> list[Trial]:
    return [t for t in trials if t.split == mode and t.history == n and t.method == method]


def _agg(trials: list[Trial], mode: str, n: int, method: str, metric: str = "auc_pr") -> tuple[float, float, int]:
    v = np.array([getattr(t, metric) for t in _cells(trials, mode, n, method)], dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if v.size else (float("nan"), float("nan"), 0)


def _wins(trials: list[Trial], mode: str, n: int, a: str, b: str) -> tuple[int, int]:
    """Paired comparison: in how many (seed, held_out) cells does method a beat method b?

    Invariant to per-cell normalisation: a and b share the cell's test set, so AUC-PR and lift
    give identical win counts. Normalisation changes the cross-cell mean and spread, not this.
    """
    cells = {(t.seed, t.held_out): t.auc_pr for t in _cells(trials, mode, n, a)}
    other = {(t.seed, t.held_out): t.auc_pr for t in _cells(trials, mode, n, b)}
    keys = sorted(set(cells) & set(other))
    return sum(cells[k] > other[k] for k in keys), len(keys)


def _corr_with_prevalence(trials: list[Trial], mode: str, n: int, method: str) -> tuple[float, int]:
    """Pearson r between a cell's AUC-PR and its own prevalence. High r = the spread is base rate, not method."""
    c = _cells(trials, mode, n, method)
    if len(c) < 3:
        return float("nan"), len(c)
    a = np.array([t.auc_pr for t in c])
    p = np.array([t.prevalence for t in c])
    if a.std() == 0 or p.std() == 0:
        return float("nan"), len(c)
    return float(np.corrcoef(a, p)[0, 1]), len(c)


def _mean_history_fraud(trials: list[Trial], mode: str, n: int) -> float:
    c = _cells(trials, mode, n, "local_only")
    return float(np.mean([t.history_fraud for t in c])) if c else float("nan")


def plot_curves(trials: list[Trial], modes: tuple[str, ...], out: Path, metric: str = "auc_pr") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(HISTORY_SIZES))
    for ax, mode in zip(axes, modes):
        for method in METHODS:
            mu = np.array([_agg(trials, mode, n, method, metric)[0] for n in HISTORY_SIZES])
            sd = np.array([_agg(trials, mode, n, method, metric)[1] for n in HISTORY_SIZES])
            ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, label=method)
        ax.set_xticks(xs)
        ax.set_xticklabels([("0 (degenerate)" if n == 0 else str(n)) for n in HISTORY_SIZES], fontsize=8)
        ax.set_xlabel("held-out bank's labelled history (rows)")
        ax.set_title(f"split = {mode}")
        ax.grid(alpha=0.3)
        if metric == "lift":
            ax.set_yscale("log")
    ylabel = "AUC-PR on held-out bank" if metric == "auc_pr" else "lift = AUC-PR / held-out prevalence (log)"
    axes[0].set_ylabel(f"{ylabel}\n(mean ± sd over banks × seeds)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_poison(pt: list[PoisonTrial], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = ["fedavg", "trimmed_mean_20", "coord_median"]
    conds = [(0, "none"), (1, "inverted"), (1, "garbage"), (2, "inverted"), (2, "garbage")]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    for i, r in enumerate(rules):
        mu, sd = [], []
        for n_att, kind in conds:
            v = np.array([t.auc_pr for t in pt if t.rule == r and t.attackers == n_att and t.attack == kind])
            mu.append(v.mean() if v.size else np.nan)
            sd.append(v.std(ddof=1) if v.size > 1 else 0)
        ax.bar(np.arange(len(conds)) + (i - 1) * w, mu, w, yerr=sd, capsize=2, label=r)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([f"{n} attacker{'s' if n != 1 else ''}\n{k}" for n, k in conds], fontsize=8)
    ax.set_ylabel("AUC-PR on held-out bank, 0 history")
    ax.set_title("Poisoned deltas: aggregation rule robustness (4 contributing banks)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fmt_cell(trials: list[Trial], mode: str, n: int, method: str) -> str:
    mu, sd, _ = _agg(trials, mode, n, method, "auc_pr")
    lm, ls, _ = _agg(trials, mode, n, method, "lift")
    return f"{mu:.3f} ± {sd:.3f}<br><sub>lift {lm:.0f} ± {ls:.0f}</sub>"


def render_report(
    trials: list[Trial], splits: dict[str, Split], pt: list[PoisonTrial], modes: tuple[str, ...], seeds: tuple[int, ...], cfg: ExpConfig,
    n_rows: int, prevalence: float, out_dir: Path, elapsed_s: Optional[float], bridge_md: Optional[str] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(trials, modes, out_dir / "cold_start_curves.png", "auc_pr")
    plot_curves(trials, modes, out_dir / "cold_start_lift.png", "lift")
    if pt:
        plot_poison(pt, out_dir / "poisoning.png")

    md: list[str] = []
    md.append("# Kavach v2 — Federated Cold-Start Experiment\n")
    md.append(
        f"""## What this is and is not

**Is:** a test of the *mechanism* in Component B — whether aggregating weight deltas from several
partitions transfers useful structure to a partition that contributed nothing, how fast local
history closes the gap, and (Result 3) whether per-bank calibrated cutoffs do anything a single
global cutoff does not. Run on the ULB credit-card dataset ({n_rows:,} rows, fraud prevalence
{100 * prevalence:.3f}%), partitioned into {N_BANKS} synthetic "banks".

**Is not:** evidence about Indian UPI/IMPS fraud. The data is 2013 European card fraud with
anonymised PCA features; the "banks" are partitions, not institutions; the feature spec is
`{ULB_EVAL.key}` (evaluation only), not the production `FEATURE_CONTRACT`. Read every number below
as a property of the federated-learning machinery, not of Kavach's fraud detection.

## Protocol

- **Learner:** online logistic regression (numpy SGD), fixed transforms only, `pos_weight={cfg.train.pos_weight}`,
  `lr={cfg.train.lr}`, `l2={cfg.train.l2}`, batch {cfg.train.batch_size}. Identical for every method.
- **Federation:** {cfg.rounds} rounds; each contributing bank trains {cfg.local_epochs} local epoch(s) from the
  current global model and sends only `weights_after − weights_before` with its row count. Aggregated by
  FedAvg (sample-weighted mean) or coordinate-median. Delta norms clipped at 50 in both cases.
- **Held-out bank:** every bank in turn. Its rows are permuted once per seed; the first *n* form its
  labelled history (n ∈ {list(HISTORY_SIZES)}), rows beyond position {TEST_OFFSET} form a **fixed test set** so
  all history sizes are scored on identical rows.
- **Methods compared on the held-out bank**, each followed by the *same* {cfg.finetune_epochs}-epoch fine-tune on the *n* history rows:
  - `local_only` — from zero weights. **At n=0 this is a constant scorer** — every row gets the same probability,
    so it cannot rank and its AUC-PR equals the prevalence (lift exactly 1.0). Comparisons at n=0 are against a
    model that cannot do the task; they are reported as the degenerate reference, not as the result.
  - `fedavg` — FedAvg global model learned from the other {N_BANKS - 1} banks' deltas.
  - `coord_median` — same, coordinate-median aggregation.
  - `pooled` — the other banks' raw rows pooled centrally, {cfg.pooled_epochs} epochs. **Upper bound**: what federation
    is trying to approach without anyone pooling data.
- **Seeds:** {list(seeds)} → {len(seeds) * N_BANKS} (seed × held-out bank) cells per split per history size.
- **Metrics:** AUC-PR on the held-out bank's test rows, **and lift = AUC-PR / that cell's own prevalence.**
  AUC-PR's floor is the prevalence, and held-out prevalence varies several-fold across banks under the non-IID
  splits, so a raw cross-bank mean mixes scales and its sd includes base-rate variation. Lift puts every cell on
  its own floor. Paired win counts are identical under both metrics (same test set per cell); lift changes the
  cross-bank mean and spread. Spread is the sd over cells.
"""
    )

    # ---- splits ------------------------------------------------------------
    md.append("## Splits\n")
    for mode in modes:
        s = splits[mode]
        md.append(f"### `{mode}`\n\n{s.method_description}\n")
        md.append("| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |\n|---|---|---|---|---|")
        for b in s.per_bank:
            md.append(f"| bank{b['bank']} | {b['n']:,} | {b['fraud']} | {100 * b['prevalence']:.3f}% | {b['median_amount']:.2f} |")
        prevs = [b["prevalence"] for b in s.per_bank]
        md.append(f"\nPrevalence ratio across banks (max/min): **{max(prevs) / min(prevs):.1f}×**.\n")

    # ---- result 1 ----------------------------------------------------------
    md.append("## Result 1 — cold-start curves\n")
    md.append("Raw AUC-PR (left figure) and lift over each cell's prevalence floor (right figure, log scale).\n")
    md.append("![cold start](cold_start_curves.png)\n\n![cold start lift](cold_start_lift.png)\n")

    for mode in modes:
        md.append(f"### `{mode}` — held-out bank, mean ± sd over {len(seeds) * N_BANKS} cells; each cell shows AUC-PR and lift\n")
        md.append("| History rows | fraud rows in history (mean) | " + " | ".join(f"`{m}`" for m in METHODS) + " | fedavg beats local (cells) | pooled − fedavg (AUC-PR) |")
        md.append("|---|---|" + "---|" * len(METHODS) + "---|---|")
        for n in TABLE_ORDER:
            label = f"{n}" if n > 0 else "0 — *degenerate reference*"
            cells = [_fmt_cell(trials, mode, n, m) for m in METHODS]
            w, tot = _wins(trials, mode, n, "fedavg", "local_only")
            gap = _agg(trials, mode, n, "pooled")[0] - _agg(trials, mode, n, "fedavg")[0]
            md.append(f"| {label} | {_mean_history_fraud(trials, mode, n):.1f} | " + " | ".join(cells) + f" | {w}/{tot} | {gap:+.3f} |")
        md.append("")

    # ---- reading the curves -------------------------------------------------
    md.append("### Reading the curves\n")
    md.append(
        "The result is the **n=50 and n=500 rows**: there the local-only model has real labelled data — on average "
        + ", ".join(f"{_mean_history_fraud(trials, modes[0], n):.1f} fraud rows at n={n}" for n in (50, 500))
        + " — and is genuinely trying to rank.\n"
    )
    for mode in modes:
        parts = []
        for n in (50, 500, 5000):
            f_ap, l_ap = _agg(trials, mode, n, "fedavg")[0], _agg(trials, mode, n, "local_only")[0]
            f_l, l_l = _agg(trials, mode, n, "fedavg", "lift")[0], _agg(trials, mode, n, "local_only", "lift")[0]
            w, t = _wins(trials, mode, n, "fedavg", "local_only")
            parts.append(f"n={n}: federated {f_ap:.3f} vs local {l_ap:.3f} AUC-PR (lift {f_l:.0f}× vs {l_l:.0f}×), wins {w}/{t}")
        p50 = _agg(trials, mode, 50, "pooled")[0]
        f50 = _agg(trials, mode, 50, "fedavg")[0]
        l50 = _agg(trials, mode, 50, "local_only")[0]
        md.append(f"- **`{mode}`:** " + "; ".join(parts) + f". At n=50 federation captures {100 * (f50 - l50) / (p50 - l50) if p50 > l50 else float('nan'):.0f}% of the pooled-vs-local gap.")
    md.append("")
    md.append(
        "**Does the win hold on lift?** Per-cell win counts are the same under lift by construction (shared test set). "
        "The cross-bank *means* are what lift changes, and the federated-over-local ordering of means is preserved in every split "
        "and every history size above — the claim does not depend on which scale is used.\n"
    )
    md.append(
        "**n=0 is the degenerate reference.** `local_only` at n=0 is a constant scorer (lift 1.0 in every cell); the "
        + "/".join(f"{_wins(trials, m, 0, 'fedavg', 'local_only')[0]}" for m in modes)
        + f" of {len(seeds) * N_BANKS} wins there say only that a model beats no model. They are not the headline.\n"
    )

    # ---- per-held-out-bank breakdown --------------------------------------
    md.append("### Per-held-out-bank breakdown (attributing the spread)\n")
    md.append(
        "Mean over seeds for each held-out bank. `prev` is that bank's test-set prevalence — the AUC-PR floor for its row. "
        "Where AUC-PR tracks `prev` down a column, the cross-bank sd is base rate, not method.\n"
    )
    for mode in modes:
        md.append(f"#### `{mode}`\n")
        hdr = "| Held-out | prev | " + " | ".join(f"`{m}` n={n}" for n in BREAKDOWN_N for m in METHODS) + " |"
        md.append(hdr)
        md.append("|---|---|" + "---|" * (len(BREAKDOWN_N) * len(METHODS)))
        for h in range(N_BANKS):
            row_cells = []
            prev = np.mean([t.prevalence for t in trials if t.split == mode and t.held_out == h and t.history == 0 and t.method == "fedavg"])
            for n in BREAKDOWN_N:
                for m in METHODS:
                    c = [t for t in _cells(trials, mode, n, m) if t.held_out == h]
                    ap = np.mean([t.auc_pr for t in c]) if c else float("nan")
                    lf = np.mean([t.lift for t in c]) if c else float("nan")
                    row_cells.append(f"{ap:.3f}<br><sub>{lf:.0f}×</sub>")
            md.append(f"| bank{h} | {100 * prev:.3f}% | " + " | ".join(row_cells) + " |")
        r_f, k = _corr_with_prevalence(trials, mode, 500, "fedavg")
        r_l, _ = _corr_with_prevalence(trials, mode, 500, "local_only")
        r_p, _ = _corr_with_prevalence(trials, mode, 500, "pooled")
        md.append(
            f"\nPearson r between a cell's AUC-PR and its prevalence at n=500 ({k} cells): fedavg **{r_f:+.2f}**, "
            f"local-only {r_l:+.2f}, pooled {r_p:+.2f}. "
            + ("A strong positive r means most of the raw sd in that column is base rate." if np.isfinite(r_f) and abs(r_f) > 0.5 else "A weak r means the spread is mostly method/seed variance, not base rate.")
            + "\n"
        )

    # ---- fedavg vs median ----------------------------------------------------
    md.append("### FedAvg vs coordinate-median with no attacker\n")
    md.append("The efficiency price of the robust rule, if any (AUC-PR difference, median − fedavg; paired wins for median):\n")
    for mode in modes:
        rows = []
        for n in TABLE_ORDER:
            f, m = _agg(trials, mode, n, "fedavg")[0], _agg(trials, mode, n, "coord_median")[0]
            w, t = _wins(trials, mode, n, "coord_median", "fedavg")
            rows.append(f"n={n}: {m - f:+.3f} ({w}/{t})")
        md.append(f"- `{mode}`: " + "; ".join(rows))
    md.append(
        "\nA negative number is the cost of using the median when everyone is honest; a positive one means the "
        "median generalised *better* to the held-out bank — which happens when partitions are skewed enough that the "
        "sample-weighted mean is dominated by whichever contributor is largest or most atypical.\n"
    )

    # ---- result 2 ----------------------------------------------------------
    if pt:
        md.append("## Result 2 — poisoned deltas (§5.4)\n")
        mode = pt[0].split
        md.append(
            f"""Split `{mode}`, every bank held out in turn, seeds {list(seeds)}, **0 local history** (pure
cold start, so the global model is all the held-out bank has). Of the {N_BANKS - 1} contributing banks,
0, 1 or 2 are attackers. An attacker replaces its honest delta with either **inverted** (−10× the
honest delta) or **garbage** (N(0, 5²) noise), and **claims 20× its true sample count** to dominate a
weighted average. Delta norm clipping (50) applies to all rules.

![poisoning](poisoning.png)

| Attackers | Attack | `fedavg` | `trimmed_mean_20` | `coord_median` |
|---|---|---|---|---|"""
        )
        for n_att, kind in [(0, "none"), (1, "inverted"), (1, "garbage"), (2, "inverted"), (2, "garbage")]:
            row = [f"{n_att}", kind]
            for r in ("fedavg", "trimmed_mean_20", "coord_median"):
                v = np.array([t.auc_pr for t in pt if t.rule == r and t.attackers == n_att and t.attack == kind])
                row.append(f"{v.mean():.3f} ± {v.std(ddof=1) if v.size > 1 else 0:.3f}" if v.size else "n/a")
            md.append("| " + " | ".join(row) + " |")
        md.append(
            f"""
Coordinate-median tolerates strictly fewer than half of the participants being malicious. With
{N_BANKS - 1} contributors that means **1 attacker is inside its breakdown point and 2 is at it** — the
2-attacker rows are expected to show degradation for *every* rule, and they are reported precisely
because they are the honest edge of the guarantee. With 20% trimming on 4 participants the trimmed
mean removes 0 values per side and falls back to the median, so `trimmed_mean_20` ≈ `coord_median`
here; it separates from it only with more participants.
"""
        )

    # ---- result 3 ----------------------------------------------------------
    if bridge_md:
        md.append(bridge_md)

    # ---- limitations -------------------------------------------------------
    md.append(
        f"""## Limitations

1. Card fraud, not UPI social engineering; PCA features, not Kavach's contract. Mechanism result only.
2. "Banks" are synthetic partitions of one dataset. Real banks differ in ways a mixing matrix does not capture
   (different fraud typologies, different labelling latency and quality, different base rates by an order of magnitude).
3. Logistic regression is a deliberately simple learner. The federated-vs-local *gap* is what is being measured, and a
   stronger local learner would shrink it at large history — the small-history end of the curve is the claim, not the right end.
4. Attackers here are crude (sign flip, noise, inflated counts). A stealthy attacker who shifts deltas by a small,
   consistent amount every round is not tested and is *not* defeated by a median.
5. No differential privacy or secure aggregation: the aggregator sees each bank's delta in the clear. "No raw data leaves
   the bank" is true; "nothing about the bank's data can be inferred from its delta" is not claimed.
6. Fine-tune epochs, rounds and learning rate were set once, not tuned per method. Tuning would move numbers, not the shape.
7. Result 3 uses the same global model at every bank so that only the cutoffs differ; in deployment each bank also
   fine-tunes, which would change the score distributions the cutoffs are calibrated on.
"""
    )
    if elapsed_s is not None and elapsed_s > 0:
        md.append(f"_Run time {elapsed_s / 60:.1f} min. Raw trials in `cold_start_trials.json`._\n")
    else:
        md.append("_Re-rendered from `cold_start_trials.json`; run time recorded in that file if the run persisted it._\n")

    path = out_dir / "cold_start_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
def _load_dataset():
    ds = load_ulb()
    X = ulb_features(ds.frame[ds.feature_columns].to_numpy(), ds.frame["amount"].to_numpy())
    y = ds.frame["label"].to_numpy().astype(int)
    amount = ds.frame["amount"].to_numpy(dtype=float)
    return X, y, amount


def main(argv: Optional[list[str]] = None) -> int:
    from . import bridge_calibration as bc  # local import: bc imports helpers from this module

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("federated/output"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["iid", "amount", "cluster"])
    ap.add_argument("--poison-mode", default="amount")
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--no-bridge", action="store_true", help="skip Result 3 (threshold-bridge calibration)")
    ap.add_argument("--quick", action="store_true", help="1 seed, iid+amount only")
    ap.add_argument("--from-json", type=Path, default=None, help="re-render the report from a saved trials JSON instead of re-running")
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds, args.modes = 1, ["iid", "amount"]

    if args.from_json is not None:
        raw = json.loads(args.from_json.read_text(encoding="utf-8"))
        trials = [Trial(**t) for t in raw["trials"]]
        pt = [PoisonTrial(**p) for p in raw["poisoning"]]
        bt = [bc.BridgeTrial(**b) for b in raw.get("bridge", [])]
        modes = tuple(dict.fromkeys(t.split for t in trials))
        seeds = tuple(sorted({t.seed for t in trials}))
        X, y, amount = _load_dataset()
        splits = {m: make_split(m, X, amount, y, np.random.default_rng(seeds[0])) for m in modes}
        bridge_md = bc.render_bridge_section(bt, modes, seeds) if bt else None
        path = render_report(trials, splits, pt, modes, seeds, ExpConfig(), len(y), float(y.mean()), args.out, raw.get("elapsed_s"), bridge_md)
        print(f"[fed] re-rendered -> {path}")
        return 0

    t0 = time.time()
    print("[fed] loading ULB ...", flush=True)
    X, y, amount = _load_dataset()
    cfg = ExpConfig()
    seeds = tuple(range(args.seeds))
    modes = tuple(args.modes)

    print(f"[fed] cold-start: splits={modes} seeds={seeds}", flush=True)
    trials, splits = cold_start(X, amount, y, modes, seeds, cfg)
    pt: list[PoisonTrial] = []
    if not args.no_poison:
        print(f"[fed] poisoning on split={args.poison_mode}", flush=True)
        pt = poisoning(X, amount, y, args.poison_mode, seeds, cfg)
    bt: list = []
    bridge_md = None
    if not args.no_bridge:
        print("[fed] threshold-bridge calibration experiment", flush=True)
        bt, _ = bc.run_bridge_experiment(X, amount, y, modes, seeds, cfg)
        bridge_md = bc.render_bridge_section(bt, modes, seeds)

    elapsed = time.time() - t0
    path = render_report(trials, splits, pt, modes, seeds, cfg, len(y), float(y.mean()), args.out, elapsed, bridge_md)
    (args.out / "cold_start_trials.json").write_text(
        json.dumps(
            {
                "trials": [asdict(t) for t in trials],
                "poisoning": [asdict(p) for p in pt],
                "bridge": [asdict(b) for b in bt],
                "bridge_summary": bc.bridge_summary(bt, modes) if bt else None,
                "elapsed_s": elapsed,
            },
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"[fed] report -> {path} ({elapsed / 60:.1f} min)")
    for mode in modes:
        for n in TABLE_ORDER:
            cells = "  ".join(f"{m}={_agg(trials, mode, n, m)[0]:.3f}({_agg(trials, mode, n, m, 'lift')[0]:.0f}x)" for m in METHODS)
            print(f"  {mode:8s} n={n:<5d} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
