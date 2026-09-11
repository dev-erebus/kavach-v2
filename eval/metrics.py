"""
Metrics for an ordinal (L0–L3) classifier against binary fraud labels.

The classifier emits a *level*, not a probability, so "precision" needs a
decision rule. We report it at every operating point a bank could actually
choose — "act on L1+", "act on L2+", "act on L3" — because those are the
only three ways the pipeline can be wired. Each operating point is a binary
classifier and gets the usual confusion matrix / precision / recall / F1.

AUC-PR uses the level itself as the ranking score. With only four distinct
scores the PR curve has at most four corners; that coarseness is a property of
the rule pipeline, not of the metric, and the report says so. When a
continuous score is available (Component B's probabilities, the reference
ML baseline) pass it as ``score`` and the curve becomes meaningful.

Nothing in this module knows about the dataset's domain. That's the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from core.levels import ThreatLevel


@dataclass
class BinaryPoint:
    """Metrics for the decision 'flag if level >= threshold'."""

    threshold: ThreatLevel
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) and not np.isnan(p) else 0.0

    @property
    def flag_rate(self) -> float:
        """Share of ALL transactions that would be actioned. The ops-load number."""
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.fp) / n if n else float("nan")

    @property
    def false_positive_rate(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else float("nan")


@dataclass
class LevelDistribution:
    """How many legit / fraud rows landed at each exact level."""

    legit: dict[ThreatLevel, int]
    fraud: dict[ThreatLevel, int]


@dataclass
class EvalMetrics:
    n: int
    n_fraud: int
    prevalence: float
    points: list[BinaryPoint]
    distribution: LevelDistribution
    auc_pr: float
    auc_pr_baseline: float  # = prevalence; a random ranker's expected AUC-PR
    auc_roc: Optional[float]
    pr_curve: tuple[np.ndarray, np.ndarray]  # (precision, recall) for plotting
    score_is_ordinal: bool
    rule_attribution: dict[str, dict[str, int]] = field(default_factory=dict)  # rule → {"tp":, "fp":}

    def point(self, level: ThreatLevel) -> BinaryPoint:
        return next(p for p in self.points if p.threshold is level)


def compute(
    y_true: np.ndarray,
    levels: Sequence[ThreatLevel] | np.ndarray,
    score: Optional[np.ndarray] = None,
    rule_names: Optional[Sequence[str]] = None,
) -> EvalMetrics:
    y = np.asarray(y_true).astype(int)
    lv = np.asarray([int(l) for l in levels], dtype=int)
    if score is None:
        score = lv.astype(float)
        ordinal = True
    else:
        score = np.asarray(score, dtype=float)
        ordinal = False

    points: list[BinaryPoint] = []
    for thr in (ThreatLevel.L1, ThreatLevel.L2, ThreatLevel.L3):
        pred = (lv >= int(thr)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        points.append(BinaryPoint(thr, int(tp), int(fp), int(fn), int(tn)))

    dist = LevelDistribution(
        legit={L: int(((lv == int(L)) & (y == 0)).sum()) for L in ThreatLevel},
        fraud={L: int(((lv == int(L)) & (y == 1)).sum()) for L in ThreatLevel},
    )

    prevalence = float(y.mean()) if len(y) else float("nan")
    auc_pr = float(average_precision_score(y, score)) if y.any() else float("nan")
    auc_roc = float(roc_auc_score(y, score)) if y.any() and not y.all() else None
    prec, rec, _ = precision_recall_curve(y, score)

    attribution: dict[str, dict[str, int]] = {}
    if rule_names is not None:
        names = np.asarray(rule_names)
        for name in sorted(set(names.tolist())):
            m = names == name
            attribution[name] = {
                "n": int(m.sum()),
                "fraud": int((m & (y == 1)).sum()),
                "legit": int((m & (y == 0)).sum()),
            }

    return EvalMetrics(
        n=int(len(y)),
        n_fraud=int(y.sum()),
        prevalence=prevalence,
        points=points,
        distribution=dist,
        auc_pr=auc_pr,
        auc_pr_baseline=prevalence,
        auc_roc=auc_roc,
        pr_curve=(prec, rec),
        score_is_ordinal=ordinal,
        rule_attribution=attribution,
    )
