"""
Feature specs — what a weight vector *means*.

The learner, aggregator and model store are deliberately ignorant of banking.
They operate on ``(name, version, dim)`` and refuse to mix vectors whose spec
differs. Two specs are registered:

* ``KAVACH_PRODUCTION`` — derived directly from ``adapters.schema.FEATURE_CONTRACT``.
  This is the one the deployed pipeline uses. Its shape is fixed by the schema
  layer (see the contract's docstring for why), so nothing here can drift from it.
* ``ULB_EVAL`` — the public credit-card dataset's own features (V1..V28 +
  log-amount). Exists only so the cold-start experiment can be run on real
  fraud labels. It is **not** a production spec and is marked as such.

Any two participants in a federation MUST share a spec, and the aggregator
enforces it. A bank on contract v2 cannot send a delta into a v1 federation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from adapters.schema import FEATURE_CONTRACT


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    names: tuple[str, ...]
    production: bool = False
    description: str = ""

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def key(self) -> str:
        """Identity string stored with every weight vector; compared on every load/aggregate."""
        return f"{self.name}@v{self.version}[{self.dim}]"

    def check(self, vec: np.ndarray, what: str = "vector") -> None:
        v = np.asarray(vec)
        if v.ndim != 1 or v.shape[0] != self.dim:
            raise SpecMismatch(f"{what} has shape {v.shape}, spec {self.key} expects ({self.dim},)")
        if not np.all(np.isfinite(v)):
            raise SpecMismatch(f"{what} contains non-finite values")


class SpecMismatch(ValueError):
    """Raised whenever a vector or stored model doesn't match the spec in force."""


# --------------------------------------------------------------------------- #
# Registered specs
# --------------------------------------------------------------------------- #
KAVACH_PRODUCTION = FeatureSpec(
    name="kavach_transaction",
    version=FEATURE_CONTRACT.version,
    names=FEATURE_CONTRACT.names,
    production=True,
    description="adapters.schema.FEATURE_CONTRACT — the deployed Kavach feature vector",
)

ULB_EVAL = FeatureSpec(
    name="ulb_creditcard_eval",
    version=1,
    names=tuple(f"V{i}" for i in range(1, 29)) + ("log1p_amount",),
    production=False,
    description="ULB credit-card PCA components + log1p(amount). EVALUATION ONLY — not a banking feature set.",
)

REGISTRY: dict[str, FeatureSpec] = {s.key: s for s in (KAVACH_PRODUCTION, ULB_EVAL)}


def ulb_features(frame_features: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Build ULB_EVAL vectors. The only transform is the contract-fixed log1p on amount — nothing fitted."""
    X = np.column_stack([np.asarray(frame_features, dtype=float), np.log1p(np.asarray(amount, dtype=float))])
    if X.shape[1] != ULB_EVAL.dim:
        raise SpecMismatch(f"expected {ULB_EVAL.dim} columns, got {X.shape[1]}")
    return X
