"""
Federated aggregation: FedAvg and coordinate-median, over weight deltas only.

Two aggregation rules, same interface:

* ``FedAvg`` — sample-weighted mean of deltas (McMahan et al., 2017). Optimal
  when every participant is honest. One poisoned participant moves the global
  model by (its weight share) × (its delta), and it can make its delta as large
  as it likes.
* ``CoordinateMedian`` — per-coordinate median of deltas (Yin et al., 2018).
  Ignores participant sample counts (a weighted median would let an attacker
  claim a huge ``n_samples``). Tolerates fewer than half the participants being
  arbitrarily malicious. The price is statistical efficiency when everyone is
  honest, and that it needs ≥3 participants to mean anything.

Brief §5.4 asks for outlier rejection on incoming deltas; ``CoordinateMedian``
*is* that, in a form with a known breakdown point, rather than a heuristic
threshold. A cheap norm-clip is applied before either rule as defence in depth
(``max_delta_norm``) — it bounds the damage of a single round, it does not fix
FedAvg.

Spec enforcement
----------------
Every delta must carry the federation's ``spec_key`` and be shaped ``dim + 1``.
Anything else is rejected before aggregation, with the bank named. This is what
makes the fixed feature contract (``adapters.schema.FEATURE_CONTRACT``) bite in
practice: a bank on a different contract version cannot silently misalign the
average — its delta is refused.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .feature_spec import FeatureSpec, SpecMismatch
from .local_trainer import Delta


@dataclass
class AggregationReport:
    rule: str
    n_received: int
    n_accepted: int
    rejected: dict[str, str]  # bank_id → reason
    clipped: list[str]  # bank_ids whose delta norm was clipped
    new_version: int


class AggregationRule(ABC):
    name: str = "abstract"

    @abstractmethod
    def combine(self, deltas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """``deltas`` shape (k, dim+1); ``weights`` shape (k,) = sample counts. Return (dim+1,)."""


class FedAvg(AggregationRule):
    name = "fedavg"

    def combine(self, deltas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        w = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1.0 / len(weights))
        return (deltas * w[:, None]).sum(axis=0)


class CoordinateMedian(AggregationRule):
    name = "coordinate_median"

    def combine(self, deltas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Weights intentionally ignored: an attacker controls its own n_samples claim.
        return np.median(deltas, axis=0)


class TrimmedMean(AggregationRule):
    """Drop the top and bottom ``trim`` fraction per coordinate, then mean. Middle ground."""

    name = "trimmed_mean"

    def __init__(self, trim: float = 0.2):
        self.trim = trim

    def combine(self, deltas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        k = deltas.shape[0]
        cut = int(np.floor(self.trim * k))
        if cut == 0 or 2 * cut >= k:
            return np.median(deltas, axis=0)
        s = np.sort(deltas, axis=0)
        return s[cut : k - cut].mean(axis=0)


class Aggregator:
    """Holds the global model for one spec and applies a rule to each round's deltas."""

    def __init__(
        self,
        spec: FeatureSpec,
        rule: AggregationRule,
        initial_params: Optional[np.ndarray] = None,
        max_delta_norm: Optional[float] = 50.0,
        min_participants: int = 1,
    ):
        self.spec = spec
        self.rule = rule
        self.max_delta_norm = max_delta_norm
        self.min_participants = min_participants
        self.global_params = np.zeros(spec.dim + 1) if initial_params is None else np.asarray(initial_params, float).copy()
        if self.global_params.shape != (spec.dim + 1,):
            raise SpecMismatch("initial_params shape does not match spec")
        self.version = 0
        self.history: list[AggregationReport] = []

    def aggregate(self, deltas: Sequence[Delta]) -> AggregationReport:
        accepted: list[Delta] = []
        rejected: dict[str, str] = {}
        clipped: list[str] = []
        for d in deltas:
            reason = self._validate(d)
            if reason:
                rejected[d.bank_id] = reason
                continue
            accepted.append(d)

        if len(accepted) < self.min_participants:
            rep = AggregationReport(self.rule.name, len(deltas), len(accepted), rejected, clipped, self.version)
            self.history.append(rep)
            return rep  # no update this round — degrade to "keep last good model"

        D = np.stack([d.delta.astype(float) for d in accepted])
        if self.max_delta_norm is not None:
            norms = np.linalg.norm(D, axis=1)
            for i, (d, nrm) in enumerate(zip(accepted, norms)):
                if nrm > self.max_delta_norm:
                    D[i] *= self.max_delta_norm / nrm
                    clipped.append(d.bank_id)
        W = np.array([max(d.n_samples, 0) for d in accepted], dtype=float)

        update = self.rule.combine(D, W)
        self.global_params = self.global_params + update
        self.version += 1
        rep = AggregationReport(self.rule.name, len(deltas), len(accepted), rejected, clipped, self.version)
        self.history.append(rep)
        return rep

    def _validate(self, d: Delta) -> Optional[str]:
        if d.spec_key != self.spec.key:
            return f"spec mismatch: delta is {d.spec_key}, federation is {self.spec.key}"
        arr = np.asarray(d.delta)
        if arr.shape != (self.spec.dim + 1,):
            return f"shape {arr.shape} != ({self.spec.dim + 1},)"
        if not np.all(np.isfinite(arr)):
            return "non-finite values"
        if d.base_version != self.version:
            return f"stale: computed against global v{d.base_version}, current is v{self.version}"
        return None
