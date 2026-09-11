"""
Markdown evaluation report with plots.

The limitations block is emitted FIRST and is not optional. A reader who
stops after the first screen must leave knowing that the dataset is
card-present / e-commerce fraud, not UPI social-engineering fraud, and that
the numbers validate pipeline mechanics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.levels import ThreatLevel  # noqa: E402
from core.rules import RuleConfig  # noqa: E402

from .dataset_loader import LabeledDataset, MappedDataset  # noqa: E402
from .metrics import EvalMetrics  # noqa: E402


@dataclass
class ReferenceBaseline:
    """A plain ML model on the dataset's own features — context, not a Kavach component."""

    name: str
    description: str
    metrics: EvalMetrics
    n_train: int
    n_test: int


@dataclass
class ReportInputs:
    dataset: LabeledDataset
    mapped: MappedDataset
    rule_config: RuleConfig
    rules_full: EvalMetrics  # v1 rules on every validated row
    rules_test: Optional[EvalMetrics]  # v1 rules on the held-out split (for comparison with baseline)
    baseline: Optional[ReferenceBaseline]
    adversarial: Optional[str] = None  # markdown section injected by §5.2 tests, if run


def _pct(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.2f}%"


def _f(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], baseline: float, out: Path) -> None:
    """Two panels: linear (what a reader expects) and log-precision (where a near-random ranker is visible at all)."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax in (ax_lin, ax_log):
        for label, (prec, rec) in curves.items():
            ax.step(rec, prec, where="post", label=label, lw=1.6)
        ax.axhline(baseline, ls="--", lw=1, color="grey", label=f"random ranker (prevalence={baseline:.4f})")
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
    ax_lin.set_ylabel("Precision")
    ax_lin.set_ylim(0, 1.02)
    ax_lin.set_title("Precision–recall (linear)")
    ax_log.set_yscale("log")
    ax_log.set_ylim(max(baseline / 3, 1e-4), 1.05)
    ax_log.set_title("Precision–recall (log precision)")
    handles, labels = ax_lin.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_level_distribution(m: EvalMetrics, out: Path) -> None:
    levels = list(ThreatLevel)
    legit = np.array([m.distribution.legit[L] for L in levels], dtype=float)
    fraud = np.array([m.distribution.fraud[L] for L in levels], dtype=float)
    # share within class, so the 0.17% fraud class is visible next to the 99.8% legit class
    legit_share = legit / legit.sum() if legit.sum() else legit
    fraud_share = fraud / fraud.sum() if fraud.sum() else fraud
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, legit_share, 0.4, label="legit (share of class)")
    ax.bar(x + 0.2, fraud_share, 0.4, label="fraud (share of class)")
    ax.set_xticks(x)
    ax.set_xticklabels([L.name for L in levels])
    ax.set_ylabel("share of class")
    ax.set_title("Where each class lands")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusions(m: EvalMetrics, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, p in zip(axes, m.points):
        mat = np.array([[p.tn, p.fp], [p.fn, p.tp]])
        ax.imshow(np.log1p(mat), cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred legit", f"pred ≥{p.threshold.name}"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["legit", "fraud"])
        ax.set_title(f"act on ≥{p.threshold.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _points_table(m: EvalMetrics) -> str:
    rows = ["| Operating point | Precision | Recall | F1 | Flag rate | FPR | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in m.points:
        rows.append(
            f"| act on ≥{p.threshold.name} | {_pct(p.precision)} | {_pct(p.recall)} | {_f(p.f1, 3)} | "
            f"{_pct(p.flag_rate)} | {_pct(p.false_positive_rate)} | {p.tp:,} | {p.fp:,} | {p.fn:,} | {p.tn:,} |"
        )
    return "\n".join(rows)


def _distribution_table(m: EvalMetrics) -> str:
    rows = ["| Level | Legit | Fraud | Fraud share at this level |", "|---|---|---|---|"]
    for L in ThreatLevel:
        lg, fr = m.distribution.legit[L], m.distribution.fraud[L]
        rows.append(f"| {L.name} ({L.action}) | {lg:,} | {fr:,} | {_pct(fr / (lg + fr)) if (lg + fr) else 'n/a'} |")
    return "\n".join(rows)


def _attribution_table(m: EvalMetrics) -> str:
    if not m.rule_attribution:
        return "_no rule attribution recorded_"
    rows = ["| Rule that fired | Rows | Fraud | Legit | Precision of rule |", "|---|---|---|---|---|"]
    for name, d in sorted(m.rule_attribution.items(), key=lambda kv: -kv[1]["n"]):
        rows.append(f"| `{name}` | {d['n']:,} | {d['fraud']:,} | {d['legit']:,} | {_pct(d['fraud'] / d['n']) if d['n'] else 'n/a'} |")
    return "\n".join(rows)


def render(inp: ReportInputs, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds, mp, m = inp.dataset, inp.mapped, inp.rules_full

    curves = {"v1 rules (ordinal level as score)": m.pr_curve}
    if inp.baseline:
        curves[f"{inp.baseline.name} (held-out split)"] = inp.baseline.metrics.pr_curve
    if inp.rules_test:
        curves["v1 rules (same held-out split)"] = inp.rules_test.pr_curve
    plot_pr_curves(curves, m.auc_pr_baseline, out_dir / "pr_curve.png")
    plot_level_distribution(m, out_dir / "level_distribution.png")
    plot_confusions(m, out_dir / "confusion_matrices.png")

    cfg = inp.rule_config
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md: list[str] = []
    md.append(f"# Kavach v2 — Evaluation Report\n\n_Generated {ts}_\n")

    # ---- LIMITATIONS FIRST -------------------------------------------------
    md.append("## ⚠️ Read this first — what these numbers do and do not show\n")
    md.append(
        f"""**This dataset is card-present / e-commerce card fraud. It is not UPI/IMPS social-engineering fraud.**

The `{ds.name}` dataset ({ds.provenance}) contains anonymised card transactions with a binary fraud
label. It has **no phone-call context, no sender age, no beneficiary-account age, no device location,
no voice-liveness signal** — none of the features Kavach's Level 2/3 logic exists to exploit. Real
Indian UPI/IMPS scam fraud (digital-arrest calls, fake-KYC, elderly-targeted mule transfers) has a
completely different signal profile and is not represented here at all.

**What this evaluation therefore validates:** that the pipeline mechanics work end to end —
adapter → canonical schema validation → rule classifier → per-level metrics — on {m.n:,} labelled rows
without silent drops, and what a v1-style amount-driven rule set does on real (if foreign-domain)
fraud labels.

**What it does not validate:** that Kavach detects the fraud it is designed for. Every number below
should be read as a *pipeline mechanics* result, not a *fraud detection* result. Do not quote these
figures as Kavach's precision or recall on Indian banking fraud.

**How the missing fields were handled.** To send rows through the real adapter/schema path, the
absent canonical fields were filled with fixed *neutral* values that cannot trigger any
social-engineering rule: sender age {mp.config.neutral_sender_age}, receiver account age
{mp.config.neutral_receiver_account_age_days} days, no active phone call, biometric verdict *unknown*.
As a consequence **the v1 rules degrade to an amount-only classifier on this data**; the rule
attribution table below confirms that only amount branches fire. Synthesising fraud-correlated
values for the missing fields would have manufactured evidence, so it was not done.

**Currency.** Amounts are in EUR and were placed on the INR rule scale with a single fixed constant
(1 EUR = ₹{mp.config.fx_to_inr}). This only positions the amount thresholds; AUC-PR and AUC-ROC are
rank-based and unaffected by it.

**Rule provenance.** The v1 thresholds below are a *reconstruction* from the build brief, because the
n8n Switch-node export was not available. They are held in `core/rules.py::RuleConfig` so the real
values can be dropped in and this report regenerated.
"""
    )

    # ---- dataset -----------------------------------------------------------
    md.append("## Dataset\n")
    md.append(f"- **Name:** `{ds.name}`\n- **Provenance:** {ds.provenance}\n- **Rows:** {ds.n:,}\n- **Fraud rows:** {ds.n_fraud:,} ({_pct(ds.prevalence)})")
    for n in ds.notes:
        md.append(f"- {n}")
    md.append("")
    md.append("### Canonical-schema validation (pipeline acceptance criterion: 100% of fields validated)\n")
    md.append(f"- Rows accepted by `KavachTransaction`: **{len(mp.transactions):,}** of {ds.n:,}")
    if mp.rejected:
        md.append(f"- Rows rejected (counted, not silently dropped): **{sum(mp.rejected.values()):,}**, of which **{mp.rejected_fraud:,} were labelled fraud** — those frauds are excluded from every metric below, which flatters recall by at most {mp.rejected_fraud}/{ds.n_fraud} of the fraud class.")
        for reason, c in mp.rejected.items():
            md.append(f"  - {c:,} × `{reason}`")
    else:
        md.append("- Rows rejected: 0")
    md.append("")

    # ---- rule config -------------------------------------------------------
    md.append("## v1 rule configuration used (reconstructed)\n")
    md.append(
        "| Parameter | Value |\n|---|---|\n"
        f"| amount_high_inr (L2 alone / L3 with signals) | ₹{cfg.amount_high_inr:,} |\n"
        f"| amount_medium_inr | ₹{cfg.amount_medium_inr:,} |\n"
        f"| amount_low_inr (L1) | ₹{cfg.amount_low_inr:,} |\n"
        f"| elderly_age | {cfg.elderly_age} |\n"
        f"| fresh_receiver_days | {cfg.fresh_receiver_days} |\n"
        f"| young_receiver_days | {cfg.young_receiver_days} |\n"
    )

    # ---- results -----------------------------------------------------------
    md.append(f"## Results — v1 rules on all {m.n:,} validated rows\n")
    md.append(f"- **AUC-PR:** {_f(m.auc_pr)} (random ranker: {_f(m.auc_pr_baseline)}; lift ×{m.auc_pr / m.auc_pr_baseline:.1f})")
    if m.auc_roc is not None:
        md.append(f"- **AUC-ROC:** {_f(m.auc_roc)}")
    if m.score_is_ordinal:
        md.append("- _Score used for ranking is the ordinal level itself (4 distinct values), so the PR curve has at most four corners. This coarseness is a property of a rule pipeline, not of the metric._")
    md.append("")
    md.append("### Per-operating-point metrics\n")
    md.append("Each row is the binary classifier 'act if level ≥ X' — the three ways the pipeline can actually be wired.\n")
    md.append(_points_table(m))
    md.append("\n![confusion matrices](confusion_matrices.png)\n")
    md.append("### Level distribution by class\n")
    md.append(_distribution_table(m))
    md.append("\n![level distribution](level_distribution.png)\n")
    md.append("### Which rule fired (false-positive breakdown)\n")
    md.append(_attribution_table(m))
    md.append("")

    # ---- baseline ----------------------------------------------------------
    if inp.baseline:
        b = inp.baseline
        md.append("## Reference point — a plain ML model on the dataset's own features\n")
        md.append(
            f"""_This is **not** a Kavach component._ It answers one question: is the rule pipeline's score
low because the data is hard, or because the rules can't see the signal? `{b.name}` — {b.description} —
is trained on {b.n_train:,} rows and evaluated on a stratified held-out split of {b.n_test:,} rows.
The v1 rules are re-scored on that same split so the comparison is like for like.

| Model | AUC-PR | AUC-ROC | Best-F1 operating point |
|---|---|---|---|
| v1 rules (act on ≥L2) | {_f(inp.rules_test.auc_pr) if inp.rules_test else 'n/a'} | {_f(inp.rules_test.auc_roc) if inp.rules_test and inp.rules_test.auc_roc is not None else 'n/a'} | {(_f(max(p.f1 for p in inp.rules_test.points), 3) if inp.rules_test else 'n/a')} |
| {b.name} | {_f(b.metrics.auc_pr)} | {_f(b.metrics.auc_roc) if b.metrics.auc_roc is not None else 'n/a'} | {_f(max(p.f1 for p in b.metrics.points), 3)} |

The gap between the two rows is the information in features the rule pipeline cannot see. On Kavach's
real domain the analogous features are the social-engineering signals — which is exactly what
Component B (federated per-bank learning) is meant to exploit, and why this baseline is the
comparison point for it.
"""
        )
        md.append("![pr curves](pr_curve.png)\n")
    else:
        md.append("![pr curve](pr_curve.png)\n")

    # ---- adversarial -------------------------------------------------------
    if inp.adversarial:
        md.append(inp.adversarial)

    # ---- limitations recap -------------------------------------------------
    md.append("## Limitations (recap)\n")
    md.append(
        """1. Wrong fraud domain: card-present/e-commerce, European cardholders, 2013. Not UPI/IMPS, not India, not social engineering.
2. Missing canonical fields filled with neutral constants; the rules therefore reduce to amount thresholds here.
3. Rule thresholds are reconstructed, not the deployed v1 export.
4. EUR→INR is a single fixed constant chosen to exercise the thresholds, not a claim about purchasing-power equivalence.
5. The dataset's PCA features are anonymised; nothing about *why* the reference model separates fraud transfers to Kavach.
6. No LLM node was exercised. Node 3's classification is out of scope for this harness (it needs a live vLLM endpoint and, more importantly, a domain-appropriate dataset).
"""
    )

    path = out_dir / "eval_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
