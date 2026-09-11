"""
Run the full evaluation: dataset → adapter → canonical schema → v1 rules → metrics → report.

    python -m eval.run_eval --dataset ulb
    python -m eval.run_eval --dataset ulb --path data/creditcard.csv
    python -m eval.run_eval --dataset ieee_cis --path data/ieee-cis/
    python -m eval.run_eval --dataset ulb --no-baseline --out eval/output

The reference ML baseline (logistic regression on the dataset's own features) is
context for the reader, not a Kavach component. It can be switched off.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.levels import ThreatLevel
from core.rules import RuleConfig, V1RuleClassifier

from . import dataset_loader, metrics, report


def classify_all(clf: V1RuleClassifier, mapped: dataset_loader.MappedDataset) -> tuple[np.ndarray, list[str]]:
    levels = np.empty(len(mapped.transactions), dtype=int)
    rules: list[str] = []
    for i, tx in enumerate(mapped.transactions):
        d = clf.classify(tx)
        levels[i] = int(d.level)
        rules.append(d.rule)
    return levels, rules


def reference_baseline(
    ds: dataset_loader.LabeledDataset,
    mapped: dataset_loader.MappedDataset,
    levels: np.ndarray,
    seed: int,
) -> tuple[report.ReferenceBaseline, metrics.EvalMetrics]:
    """Logistic regression on the dataset's own features, on a stratified held-out split.

    Returns the baseline AND the rules re-scored on the identical test rows.
    """
    X = ds.frame.loc[mapped.kept_index, ds.feature_columns + ["amount"]].to_numpy(dtype=float)
    X[:, -1] = np.log1p(X[:, -1])
    y = mapped.labels
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])[:, 1]

    # Give the baseline the same three operating points as the rules by binning
    # its probability into L0–L3 at fixed cutoffs. Cutoffs are arbitrary
    # (0.5 / 0.9 / 0.99) — the meaningful comparison is AUC-PR, which needs none.
    b_levels = np.digitize(proba, [0.5, 0.9, 0.99])
    b_metrics = metrics.compute(y[te], b_levels, score=proba)
    rules_test = metrics.compute(y[te], levels[te])

    base = report.ReferenceBaseline(
        name="logistic regression on V1..V28 + log(amount)",
        description="StandardScaler → class-balanced logistic regression, default regularisation, no tuning",
        metrics=b_metrics,
        n_train=len(tr),
        n_test=len(te),
    )
    return base, rules_test


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ulb", "ieee_cis"], default="ulb")
    ap.add_argument("--path", type=Path, default=None, help="local CSV (ulb) or directory (ieee_cis)")
    ap.add_argument("--out", type=Path, default=Path("eval/output"))
    ap.add_argument("--no-baseline", action="store_true", help="skip the reference ML baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fx", type=str, default=None, help="override FX-to-INR constant, e.g. 90")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[eval] loading {args.dataset} ...", flush=True)
    ds = dataset_loader.load(args.dataset, args.path)
    print(f"[eval] {ds.n:,} rows, {ds.n_fraud:,} fraud ({100 * ds.prevalence:.3f}%)")

    cfg = dataset_loader.EvalMappingConfig() if args.fx is None else dataset_loader.EvalMappingConfig(
        fx_to_inr=__import__("decimal").Decimal(args.fx)
    )
    print("[eval] mapping rows through EvalDatasetAdapter -> KavachTransaction ...", flush=True)
    mapped = dataset_loader.to_kavach_transactions(ds, cfg)
    print(f"[eval] accepted {len(mapped.transactions):,}, rejected {sum(mapped.rejected.values()):,} {mapped.rejected or ''}")

    rule_cfg = RuleConfig()
    clf = V1RuleClassifier(rule_cfg)
    print("[eval] classifying with v1 rules ...", flush=True)
    levels, rule_names = classify_all(clf, mapped)
    full = metrics.compute(mapped.labels, levels, rule_names=rule_names)

    baseline = rules_test = None
    if not args.no_baseline:
        print("[eval] fitting reference logistic-regression baseline ...", flush=True)
        baseline, rules_test = reference_baseline(ds, mapped, levels, args.seed)

    path = report.render(
        report.ReportInputs(ds, mapped, rule_cfg, full, rules_test, baseline),
        args.out,
    )

    # Machine-readable summary next to the markdown.
    summary = {
        "dataset": ds.name,
        "n": full.n,
        "n_fraud": full.n_fraud,
        "prevalence": full.prevalence,
        "rejected": mapped.rejected,
        "auc_pr": full.auc_pr,
        "auc_roc": full.auc_roc,
        "points": {
            p.threshold.name: {"precision": p.precision, "recall": p.recall, "f1": p.f1, "flag_rate": p.flag_rate,
                               "tp": p.tp, "fp": p.fp, "fn": p.fn, "tn": p.tn}
            for p in full.points
        },
        "rule_attribution": full.rule_attribution,
        "baseline": None if baseline is None else {
            "name": baseline.name, "auc_pr": baseline.metrics.auc_pr, "auc_roc": baseline.metrics.auc_roc,
            "rules_on_same_split_auc_pr": rules_test.auc_pr if rules_test else None,
        },
        "domain_caveat": "card-present/e-commerce fraud dataset; validates pipeline mechanics only, NOT UPI/IMPS social-engineering detection",
    }
    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"[eval] report -> {path}  ({time.time() - t0:.0f}s)")
    for p in full.points:
        print(f"  act on >={p.threshold.name}: precision {100 * p.precision:6.2f}%  recall {100 * p.recall:6.2f}%  F1 {p.f1:.3f}  flag rate {100 * p.flag_rate:.2f}%")
    print(f"  AUC-PR {full.auc_pr:.4f} (random {full.auc_pr_baseline:.4f})" + (f"   | baseline LR AUC-PR {baseline.metrics.auc_pr:.4f}" if baseline else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
