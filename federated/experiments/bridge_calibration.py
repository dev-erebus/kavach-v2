"""
Threshold-bridge experiment: do per-bank calibrated cutoffs beat one global cutoff?

This is the brief's actual Component B claim (§2.3): ``threshold_bridge.py`` replaces the
hardcoded rule branches with **per-bank calibrated cutoffs**, so that "₹1,00,000 is routine
here and alarming there" is expressed in each bank's own score distribution. Everything in
``cold_start.py`` is threshold-free ranking (AUC-PR) and never exercises that mechanism.
This module does.

Protocol
--------
For each split and seed:

1. Train ONE FedAvg global model over all 5 banks (5 rounds). Every bank scores with this
   same model, so the only thing that differs between conditions is the cutoffs. (Per-bank
   fine-tuning would confound the comparison; it is deliberately switched off here.)
2. Each bank's rows are split into thirds: a *fit* third (used only by the full-stack row
   below), a *calibration* third, and an *evaluation* third. Every strategy is scored on
   the same evaluation third.
3. Four cutoff strategies, two operating-point definitions × {global, per-bank}, plus one
   full-stack row:

   **Flag-rate targets** (unlabelled; what a bank can do on day one — "we can afford to
   review 1% of traffic at L2"):
     - ``global_flagrate``  — one set of cutoffs from the POOLED calibration scores, at the
       target rates. This is the v1 analogue: a single number for everyone.
     - ``perbank_flagrate`` — each bank's cutoffs from its OWN calibration scores.

   **Precision targets** (labelled; what a bank can do once ``/feedback`` has accumulated):
     - ``global_precision``  — cutoffs from pooled labelled calibration data.
     - ``perbank_precision`` — each bank's cutoffs from its own labelled calibration data.

   **Full stack** (the brief's actual §2.2 configuration — global baseline + local
   fine-tuning + per-bank cutoff):
     - ``fullstack_flagrate`` — each bank fine-tunes the global model on its *fit* third,
       calibrates flag-rate cutoffs on its *calibration* third with that fine-tuned model,
       and is evaluated on its *evaluation* third. The other four rows hold the model fixed
       to isolate the cutoff effect; this row tests the combination the brief describes.

4. Every strategy is evaluated on every bank's evaluation third: per level (≥L1/≥L2/≥L3)
   precision, recall, and alert volume (flag rate) — plus the cutoffs themselves, so the
   report can say how different the per-bank cutoffs actually are.

What would count as "per-bank wins"
-----------------------------------
* Under flag-rate targets: the per-bank strategy should hold each bank's alert volume at
  its budget, while the global cutoff over-alerts some banks and starves others whenever
  their score distributions differ. Total recall for the same total alert volume may
  favour EITHER: a global cutoff on comparable probabilities is the recall-optimal
  allocation of a fixed total budget (alerts go where scores are highest), while per-bank
  budgets equalise ops load. Both are reported; which one a bank wants is a policy choice.
* Under precision targets: the per-bank strategy should meet each bank's precision target
  on its own eval half; the global cutoff meets it only in aggregate.

If the two are nearly indistinguishable — expected under ``iid`` — that is the finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from core.levels import ThreatLevel
from federated.aggregator import FedAvg
from federated.feature_spec import ULB_EVAL
from federated.local_trainer import OnlineLogisticRegression
from federated.threshold_bridge import (
    Cutoffs,
    FlagRateTargets,
    calibrate_by_flag_rate,
    calibrate_by_precision,
)

from .cold_start import ExpConfig, N_BANKS, Split, make_split, run_federation

FLAG_TARGETS = FlagRateTargets(l1=0.05, l2=0.01, l3=0.001)
PRECISION_TARGETS = (0.02, 0.10, 0.50)  # ≥L1, ≥L2, ≥L3
STRATEGIES = ("global_flagrate", "perbank_flagrate", "global_precision", "perbank_precision", "fullstack_flagrate")
LEVELS = (ThreatLevel.L1, ThreatLevel.L2, ThreatLevel.L3)


@dataclass
class BridgeTrial:
    split: str
    seed: int
    bank: int
    strategy: str
    level: str  # "L1" / "L2" / "L3" meaning "act on >= level"
    cutoff: float
    n_eval: int
    n_fraud_eval: int
    alerts: int
    tp: int
    flag_rate: float
    precision: float  # nan if no alerts
    recall: float


def _eval_level(p: np.ndarray, y: np.ndarray, cutoff: float) -> tuple[int, int, float, float, float]:
    flagged = p >= cutoff
    alerts = int(flagged.sum())
    tp = int((flagged & (y == 1)).sum())
    fr = alerts / len(p) if len(p) else float("nan")
    prec = tp / alerts if alerts else float("nan")
    rec = tp / y.sum() if y.sum() else float("nan")
    return alerts, tp, fr, prec, rec


def _cut_for(c: Cutoffs, level: ThreatLevel) -> float:
    return {ThreatLevel.L1: c.l1, ThreatLevel.L2: c.l2, ThreatLevel.L3: c.l3}[level]


def run_bridge_experiment(
    X: np.ndarray, amount: np.ndarray, y: np.ndarray, modes: tuple[str, ...], seeds: tuple[int, ...], cfg: ExpConfig, log=print
) -> tuple[list[BridgeTrial], dict[str, Split]]:
    trials: list[BridgeTrial] = []
    splits: dict[str, Split] = {}
    for mode in modes:
        for seed in seeds:
            rng = np.random.default_rng(5000 + seed)
            split = make_split(mode, X, amount, y, rng)
            splits.setdefault(mode, split)

            # One global model from all banks; the same scorer everywhere.
            bank_data = {k: (X[split.bank_of_row == k], y[split.bank_of_row == k]) for k in range(N_BANKS)}
            params, _ = run_federation(FedAvg(), bank_data, cfg, seed)
            model = OnlineLogisticRegression(ULB_EVAL, cfg.train, params=params)

            cal: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            ev: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            fit: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            ev_X: dict[int, np.ndarray] = {}
            cal_X: dict[int, np.ndarray] = {}
            for k in range(N_BANKS):
                idx = np.where(split.bank_of_row == k)[0]
                perm = rng.permutation(idx)
                a, b = len(perm) // 3, 2 * len(perm) // 3
                fit[k] = (X[perm[:a]], y[perm[:a]])
                cal_X[k], ev_X[k] = X[perm[a:b]], X[perm[b:]]
                cal[k] = (model.predict_proba(cal_X[k]), y[perm[a:b]])
                ev[k] = (model.predict_proba(ev_X[k]), y[perm[b:]])

            pooled_p = np.concatenate([cal[k][0] for k in range(N_BANKS)])
            pooled_y = np.concatenate([cal[k][1] for k in range(N_BANKS)])
            global_fr = calibrate_by_flag_rate(pooled_p, FLAG_TARGETS)
            global_pr = calibrate_by_precision(pooled_p, pooled_y, PRECISION_TARGETS)

            for k in range(N_BANKS):
                pc, yc = cal[k]
                pe, ye = ev[k]
                # Full stack: this bank's live model = global + fine-tune on its fit third.
                local = OnlineLogisticRegression(ULB_EVAL, cfg.train, params=params)
                local.fit_epochs(fit[k][0], fit[k][1], np.random.default_rng(seed * 10 + k), epochs=cfg.finetune_epochs)
                fs_cuts = calibrate_by_flag_rate(local.predict_proba(cal_X[k]), FLAG_TARGETS)
                pe_local = local.predict_proba(ev_X[k])

                strategies = {
                    "global_flagrate": (global_fr, pe),
                    "perbank_flagrate": (calibrate_by_flag_rate(pc, FLAG_TARGETS), pe),
                    "global_precision": (global_pr, pe),
                    "perbank_precision": (calibrate_by_precision(pc, yc, PRECISION_TARGETS) if yc.sum() > 0 else global_pr, pe),
                    "fullstack_flagrate": (fs_cuts, pe_local),
                }
                for sname, (cuts, scores) in strategies.items():
                    for L in LEVELS:
                        c = _cut_for(cuts, L)
                        alerts, tp, fr, prec, rec = _eval_level(scores, ye, c)
                        trials.append(BridgeTrial(mode, seed, k, sname, L.name, c, len(scores), int(ye.sum()), alerts, tp, fr, prec, rec))
            log(f"  [bridge {mode} seed={seed}] done")
    return trials, splits


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _sel(trials: list[BridgeTrial], **kw) -> list[BridgeTrial]:
    return [t for t in trials if all(getattr(t, k) == v for k, v in kw.items())]


def _mean(vals) -> float:
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    return float(v.mean()) if v.size else float("nan")


def _sd(vals) -> float:
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else 0.0


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def _pct(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.{nd}f}%"


def render_bridge_section(trials: list[BridgeTrial], modes: tuple[str, ...], seeds: tuple[int, ...]) -> str:
    md: list[str] = []
    md.append("## Result 3 — threshold bridge: per-bank calibrated cutoffs vs one global cutoff\n")
    md.append(
        f"""This is the mechanism the brief actually promises for Component B (§2.3). One FedAvg global model
is trained over all {N_BANKS} banks per (split, seed); each bank's rows are split into a *fit* third, a *calibration*
third and an *evaluation* third, and every strategy is scored on the same evaluation third.

**Scope — read before the tables.** Four of the five rows use that single global model as the scorer at every bank,
so that **only the cutoffs differ** between conditions. That isolates the cutoff effect, but it means those rows test
*per-bank cutoffs*, not the brief's full "global baseline + local fine-tuning + per-bank cutoff" stack (§2.2). The
fifth row, `fullstack_flagrate`, adds the missing piece: each bank fine-tunes the global model on its fit third, then
calibrates per-bank flag-rate cutoffs with its own fine-tuned model. Where the full stack differs from
`perbank_flagrate`, the difference is the local fine-tuning, not the cutoffs.

- **Flag-rate targets** (unlabelled, day-one): ≥L1 {FLAG_TARGETS.l1:.0%}, ≥L2 {FLAG_TARGETS.l2:.0%}, ≥L3 {FLAG_TARGETS.l3:.1%} of traffic.
  `global_flagrate` = one cutoff set from pooled calibration scores (the v1 "single number" analogue);
  `perbank_flagrate` = each bank's own; `fullstack_flagrate` = each bank's own, with its fine-tuned model.
- **Precision targets** (labelled, after `/feedback`): ≥L1 {PRECISION_TARGETS[0]:.0%}, ≥L2 {PRECISION_TARGETS[1]:.0%}, ≥L3 {PRECISION_TARGETS[2]:.0%}.
  `global_precision` from pooled labelled calibration data; `perbank_precision` from each bank's own.

Cells: {N_BANKS} banks × {len(seeds)} seeds = {N_BANKS * len(seeds)} per split. Fraud counts per bank-third are small
(tens), so ≥L3 precision/recall are noisy; read ≥L2 as the main operating point.
"""
    )

    for mode in modes:
        md.append(f"### `{mode}`\n")
        # --- alert-volume adherence (flag-rate strategies) -------------------
        md.append("**Alert volume per bank at ≥L2 (target 1.00% of each bank's traffic)** — mean over seeds:\n")
        hdr = "| Bank | eval prevalence | `global_flagrate` flag rate | `perbank_flagrate` flag rate | `fullstack_flagrate` flag rate | global recall | per-bank recall | full-stack recall | global precision | per-bank precision | full-stack precision |"
        md.append(hdr)
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in range(N_BANKS):
            g = _sel(trials, split=mode, bank=k, strategy="global_flagrate", level="L2")
            p = _sel(trials, split=mode, bank=k, strategy="perbank_flagrate", level="L2")
            fs = _sel(trials, split=mode, bank=k, strategy="fullstack_flagrate", level="L2")
            prev = _mean([t.n_fraud_eval / t.n_eval for t in g])
            md.append(
                f"| bank{k} | {_pct(prev, 3)} | {_pct(_mean([t.flag_rate for t in g]))} | {_pct(_mean([t.flag_rate for t in p]))} | {_pct(_mean([t.flag_rate for t in fs]))} | "
                f"{_pct(_mean([t.recall for t in g]), 1)} | {_pct(_mean([t.recall for t in p]), 1)} | {_pct(_mean([t.recall for t in fs]), 1)} | "
                f"{_pct(_mean([t.precision for t in g]), 1)} | {_pct(_mean([t.precision for t in p]), 1)} | {_pct(_mean([t.precision for t in fs]), 1)} |"
            )
        # dispersion of global flag rate across banks vs target
        g_rates = [t.flag_rate for t in _sel(trials, split=mode, strategy="global_flagrate", level="L2")]
        p_rates = [t.flag_rate for t in _sel(trials, split=mode, strategy="perbank_flagrate", level="L2")]
        md.append("")
        md.append(
            f"Across bank-cells the ≥L2 alert volume under the global cutoff ranges **{_pct(min(g_rates))}–{_pct(max(g_rates))}** "
            f"(sd {_pct(_sd(g_rates))}) against a 1.00% budget; under per-bank calibration it ranges "
            f"**{_pct(min(p_rates))}–{_pct(max(p_rates))}** (sd {_pct(_sd(p_rates))}). "
            "The per-bank residual is calibration-vs-evaluation sampling noise; the global spread is the distribution shift between banks."
        )
        md.append("")

        # --- total recall for the same total alerts ---------------------------
        md.append("**Totals over all banks, per level (same model, same overall alert budget):**\n")
        md.append("| Level | Strategy | total alerts / seed | total TP / seed | pooled precision | pooled recall |")
        md.append("|---|---|---|---|---|---|")
        for L in ("L1", "L2", "L3"):
            for s in ("global_flagrate", "perbank_flagrate", "fullstack_flagrate"):
                rows = _sel(trials, split=mode, strategy=s, level=L)
                per_seed_alerts = [sum(t.alerts for t in rows if t.seed == sd) for sd in seeds]
                per_seed_tp = [sum(t.tp for t in rows if t.seed == sd) for sd in seeds]
                per_seed_fraud = [sum(t.n_fraud_eval for t in rows if t.seed == sd) for sd in seeds]
                pp = _mean([tp / a if a else np.nan for tp, a in zip(per_seed_tp, per_seed_alerts)])
                pr = _mean([tp / f if f else np.nan for tp, f in zip(per_seed_tp, per_seed_fraud)])
                md.append(f"| ≥{L} | `{s}` | {_mean(per_seed_alerts):,.0f} | {_mean(per_seed_tp):.1f} | {_pct(pp, 1)} | {_pct(pr, 1)} |")
        md.append("")

        # --- precision-target strategies ---------------------------------------
        md.append(f"**Precision targets — achieved ≥L2 precision per bank (target {PRECISION_TARGETS[1]:.0%}), mean over seeds:**\n")
        md.append("| Bank | `global_precision` precision | `perbank_precision` precision | global recall | per-bank recall | global alerts | per-bank alerts |")
        md.append("|---|---|---|---|---|---|---|")
        for k in range(N_BANKS):
            g = _sel(trials, split=mode, bank=k, strategy="global_precision", level="L2")
            p = _sel(trials, split=mode, bank=k, strategy="perbank_precision", level="L2")
            md.append(
                f"| bank{k} | {_pct(_mean([t.precision for t in g]), 1)} | {_pct(_mean([t.precision for t in p]), 1)} | "
                f"{_pct(_mean([t.recall for t in g]), 1)} | {_pct(_mean([t.recall for t in p]), 1)} | "
                f"{_mean([t.alerts for t in g]):.0f} | {_mean([t.alerts for t in p]):.0f} |"
            )
        g_hit = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, split=mode, strategy="global_precision", level="L2") if not np.isnan(t.precision)]
        p_hit = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, split=mode, strategy="perbank_precision", level="L2") if not np.isnan(t.precision)]
        md.append("")
        md.append(
            f"Bank-cells meeting the ≥L2 precision target on their evaluation half: global cutoff **{sum(g_hit)}/{len(g_hit)}**, "
            f"per-bank cutoff **{sum(p_hit)}/{len(p_hit)}**."
        )
        md.append("")

        # --- how different are the cutoffs? ------------------------------------
        md.append("**How different are the per-bank cutoffs?** (probability cutoff for ≥L2, mean over seeds; global cutoff for reference)\n")
        md.append("| | " + " | ".join(f"bank{k}" for k in range(N_BANKS)) + " | global | max/min ratio |")
        md.append("|---|" + "---|" * (N_BANKS + 2))
        for s_pb, s_g in (("perbank_flagrate", "global_flagrate"), ("fullstack_flagrate", "global_flagrate"), ("perbank_precision", "global_precision")):
            cuts = [_mean([t.cutoff for t in _sel(trials, split=mode, bank=k, strategy=s_pb, level="L2")]) for k in range(N_BANKS)]
            gcut = _mean([t.cutoff for t in _sel(trials, split=mode, strategy=s_g, level="L2")])
            ratio = max(cuts) / min(cuts) if min(cuts) > 0 else float("nan")
            md.append(f"| `{s_pb}` | " + " | ".join(f"{c:.4f}" for c in cuts) + f" | {gcut:.4f} | {ratio:.1f}× |")
        md.append("")

    # ---- reading, computed --------------------------------------------------
    md.append("### Reading Result 3\n")
    for mode in modes:
        summ = bridge_summary(trials, (mode,))[mode]
        g_rng = (summ["l2_flag_rate_global_min"], summ["l2_flag_rate_global_max"])
        p_rng = (summ["l2_flag_rate_perbank_min"], summ["l2_flag_rate_perbank_max"])
        d_recall = summ["l2_pooled_recall_perbank"] - summ["l2_pooled_recall_global"]
        g_hit = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, split=mode, strategy="global_precision", level="L2") if not np.isnan(t.precision)]
        p_hit = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, split=mode, strategy="perbank_precision", level="L2") if not np.isnan(t.precision)]
        cuts = [_mean([t.cutoff for t in _sel(trials, split=mode, bank=k, strategy="perbank_flagrate", level="L2")]) for k in range(N_BANKS)]
        ratio = max(cuts) / min(cuts) if min(cuts) > 0 else float("nan")
        md.append(
            f"- **`{mode}`:** per-bank >=L2 cutoffs differ by **{ratio:.1f}x** across banks. A single global cutoff sends "
            f"**{_pct(g_rng[0])}-{_pct(g_rng[1])}** of each bank's traffic to review against a 1% budget; per-bank calibration holds it at "
            f"**{_pct(p_rng[0])}-{_pct(p_rng[1])}**. For the same total alert volume, pooled >=L2 recall changes by **{100 * d_recall:+.1f} pts** "
            f"(per-bank minus global). Labelled precision-target calibration meets its >=L2 target in {sum(g_hit)}/{len(g_hit)} bank-cells with the "
            f"global cutoff vs {sum(p_hit)}/{len(p_hit)} per-bank."
        )
        fs_rng = ( min(t.flag_rate for t in _sel(trials, split=mode, strategy="fullstack_flagrate", level="L2")),
                   max(t.flag_rate for t in _sel(trials, split=mode, strategy="fullstack_flagrate", level="L2")) )
        seeds_ = sorted({t.seed for t in trials})
        def _pooled_recall(strategy):
            rows = _sel(trials, split=mode, strategy=strategy, level="L2")
            return _mean([sum(t.tp for t in rows if t.seed == sd) / max(1, sum(t.n_fraud_eval for t in rows if t.seed == sd)) for sd in seeds_])
        md.append(
            f"  Full stack (`fullstack_flagrate`, global + local fine-tune + per-bank cutoff): alert volume "
            f"**{_pct(fs_rng[0])}-{_pct(fs_rng[1])}**, pooled >=L2 recall {_pct(_pooled_recall('fullstack_flagrate'), 1)} vs "
            f"{_pct(_pooled_recall('perbank_flagrate'), 1)} for per-bank cutoffs on the shared model and {_pct(_pooled_recall('global_flagrate'), 1)} for the global cutoff."
        )
    md.append("")
    # Cross-split totals so the prose below is computed, not asserted.
    d_recalls = {m: bridge_summary(trials, (m,))[m]["l2_pooled_recall_perbank"] - bridge_summary(trials, (m,))[m]["l2_pooled_recall_global"] for m in modes}
    g_hits = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, strategy="global_precision", level="L2") if not np.isnan(t.precision)]
    p_hits = [t.precision >= PRECISION_TARGETS[1] for t in _sel(trials, strategy="perbank_precision", level="L2") if not np.isnan(t.precision)]
    worst = min(d_recalls, key=d_recalls.get)
    md.append(
        f"""**What the bridge demonstrably buys, on this data:** *alert-budget adherence per bank.* Under the non-IID splits a
single cutoff over-alerts the bank whose score distribution sits highest and starves the others - a 1% budget becomes
several percent at one bank and a fraction of a percent at another. Per-bank flag-rate calibration removes that, with no
labels needed.

**What it does not buy here:** *more fraud caught.* For the same total number of alerts, pooled >=L2 recall under per-bank
budgets minus recall under the global cutoff is {", ".join(f"{100 * d_recalls[m]:+.1f} pts (`{m}`)" for m in modes)}. Per-bank budgets
never gain more than {100 * max(d_recalls.values()):+.1f} pts and give up {100 * -d_recalls[worst]:.1f} pts under `{worst}`. That is expected from the construction: the global model's
probabilities are comparable across banks, so a single cutoff already allocates alerts to the highest-scoring rows regardless
of bank - the recall-optimal allocation of a fixed total budget. Per-bank budgets trade some of that optimality for predictable
ops load at every bank. Which one a federation wants is a policy decision, not a modelling one, and the bridge supports both.

**Labelled per-bank precision calibration is unreliable at this fraud volume.** Across all splits the >=L2 precision target
is met on the evaluation half in {sum(g_hits)}/{len(g_hits)} bank-cells with the global cutoff and {sum(p_hits)}/{len(p_hits)} with per-bank cutoffs, and
which strategy does better flips by split. With tens of confirmed frauds per bank-half, a cutoff fitted to one half does not
hold on the other whichever way it is fitted. Precision-target calibration should wait until `/feedback` has accumulated
hundreds of confirmed outcomes per bank; until then the bridge should run on flag-rate calibration. Under `iid` the flag-rate
strategies are, as expected, indistinguishable - that row is the control.
"""
    )
    return "\n".join(md)


def bridge_summary(trials: list[BridgeTrial], modes: tuple[str, ...]) -> dict:
    """Compact numbers for the README / JSON: ≥L2 alert-volume spread and pooled recall, per split."""
    out = {}
    for mode in modes:
        g = [t.flag_rate for t in _sel(trials, split=mode, strategy="global_flagrate", level="L2")]
        p = [t.flag_rate for t in _sel(trials, split=mode, strategy="perbank_flagrate", level="L2")]
        seeds = sorted({t.seed for t in trials})
        def pooled_recall(s):
            rows = _sel(trials, split=mode, strategy=s, level="L2")
            return _mean([sum(t.tp for t in rows if t.seed == sd) / max(1, sum(t.n_fraud_eval for t in rows if t.seed == sd)) for sd in seeds])
        out[mode] = {
            "l2_flag_rate_global_min": min(g), "l2_flag_rate_global_max": max(g),
            "l2_flag_rate_perbank_min": min(p), "l2_flag_rate_perbank_max": max(p),
            "l2_pooled_recall_global": pooled_recall("global_flagrate"),
            "l2_pooled_recall_perbank": pooled_recall("perbank_flagrate"),
            "l2_pooled_recall_fullstack": pooled_recall("fullstack_flagrate"),
            "l2_flag_rate_fullstack_min": min(t.flag_rate for t in _sel(trials, split=mode, strategy="fullstack_flagrate", level="L2")),
            "l2_flag_rate_fullstack_max": max(t.flag_rate for t in _sel(trials, split=mode, strategy="fullstack_flagrate", level="L2")),
        }
    return out
