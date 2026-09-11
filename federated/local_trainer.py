"""
Per-bank online learner.

Plain logistic regression trained by mini-batch SGD, in numpy, with no fitted
preprocessing — every transform is fixed in the ``FeatureSpec`` (see
``feature_spec.py`` for why a per-bank scaler would break federated averaging).

What leaves the bank
--------------------
``LocalTrainer.export_delta()`` returns ``weights_now − weights_at_last_sync``
plus the sample count. That is the ONLY thing the aggregator ever sees. No
feature vector, no label, no transaction id crosses the boundary — the
``Delta`` dataclass has no field that could carry one.

Why logistic regression
-----------------------
The brief asks for something lightweight and honest. LR gives a calibratable
probability the threshold bridge can map to L0–L3, its weights average
meaningfully under FedAvg (the same cannot be said of trees), and a
coordinate-wise median over its weights is well-defined. It is also small
enough that a bank can retrain it on every ``/feedback`` event.

Class imbalance
---------------
Fraud prevalence is ~0.1–0.5%. ``pos_weight`` up-weights the positive class
in the loss; it is a spec-level hyperparameter shared by all participants
(a per-bank value would, again, change what "the same weights" mean).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .feature_spec import FeatureSpec, SpecMismatch


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.05
    l2: float = 1e-4
    batch_size: int = 256
    epochs: int = 3
    pos_weight: float = 50.0  # loss weight on fraud rows; shared across participants
    clip_grad: float = 5.0  # per-batch gradient norm clip; keeps a garbage batch from blowing up weights


@dataclass
class Delta:
    """The one object that crosses the bank boundary. Deliberately carries no data."""

    bank_id: str
    spec_key: str
    base_version: int  # global model version this delta was computed against
    delta: np.ndarray  # shape (dim + 1,) — weights then bias
    n_samples: int  # how many labelled rows contributed; FedAvg weight


class OnlineLogisticRegression:
    """Weights + bias, SGD updates, predict_proba. Nothing else."""

    def __init__(self, spec: FeatureSpec, cfg: TrainConfig = TrainConfig(), params: Optional[np.ndarray] = None):
        self.spec = spec
        self.cfg = cfg
        if params is None:
            params = np.zeros(spec.dim + 1)
        self.set_params(params)
        self.n_seen = 0

    # ---- params ------------------------------------------------------------
    def get_params(self) -> np.ndarray:
        return self._params.copy()

    def set_params(self, params: np.ndarray) -> None:
        p = np.asarray(params, dtype=float)
        if p.shape != (self.spec.dim + 1,):
            raise SpecMismatch(f"params shape {p.shape}, spec {self.spec.key} expects ({self.spec.dim + 1},)")
        if not np.all(np.isfinite(p)):
            raise SpecMismatch("params contain non-finite values")
        self._params = p.copy()

    @property
    def w(self) -> np.ndarray:
        return self._params[:-1]

    @property
    def b(self) -> float:
        return float(self._params[-1])

    # ---- inference ---------------------------------------------------------
    def decision(self, X: np.ndarray) -> np.ndarray:
        X = self._check_X(X)
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision(X)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    # ---- learning ----------------------------------------------------------
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """One SGD step on a batch. Weighted log-loss + L2."""
        X = self._check_X(X)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y length mismatch")
        p = self.predict_proba(X)
        sw = np.where(y == 1.0, self.cfg.pos_weight, 1.0)
        err = (p - y) * sw
        gw = X.T @ err / X.shape[0] + self.cfg.l2 * self.w
        gb = float(err.mean())
        g = np.append(gw, gb)
        norm = np.linalg.norm(g)
        if norm > self.cfg.clip_grad:
            g *= self.cfg.clip_grad / norm
        self._params = self._params - self.cfg.lr * g
        self.n_seen += X.shape[0]

    def fit_epochs(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, epochs: Optional[int] = None) -> None:
        """Shuffled mini-batch passes. Used for local rounds in the federation and for the experiments."""
        X = self._check_X(X)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if n == 0:
            return
        for _ in range(epochs if epochs is not None else self.cfg.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.cfg.batch_size):
                idx = order[s : s + self.cfg.batch_size]
                self.partial_fit(X[idx], y[idx])

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.spec.dim:
            raise SpecMismatch(f"X has {X.shape[1]} columns, spec {self.spec.key} expects {self.spec.dim}")
        return X


class LocalTrainer:
    """A bank's local learner: global baseline + local fine-tuning, and delta export.

    Lifecycle in the federation:
        1. ``sync(global_params, version)``   — adopt the aggregator's latest baseline
        2. ``learn(X, y)`` (from /feedback)   — fine-tune locally, any number of times
        3. ``export_delta()``                 — hand ONLY the weight movement to the aggregator
        4. back to 1 on the next round
    """

    def __init__(self, bank_id: str, spec: FeatureSpec, cfg: TrainConfig = TrainConfig(), seed: int = 0):
        self.bank_id = bank_id
        self.spec = spec
        self.model = OnlineLogisticRegression(spec, cfg)
        self._rng = np.random.default_rng(seed)
        self._sync_params = self.model.get_params()
        self._sync_version = 0
        self._n_since_sync = 0

    def sync(self, global_params: np.ndarray, global_version: int) -> None:
        self.model.set_params(global_params)
        self._sync_params = self.model.get_params()
        self._sync_version = global_version
        self._n_since_sync = 0

    def learn(self, X: np.ndarray, y: np.ndarray, epochs: Optional[int] = None) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        self.model.fit_epochs(X, y, self._rng, epochs=epochs)
        self._n_since_sync += X.shape[0]

    def export_delta(self) -> Delta:
        return Delta(
            bank_id=self.bank_id,
            spec_key=self.spec.key,
            base_version=self._sync_version,
            delta=self.model.get_params() - self._sync_params,
            n_samples=self._n_since_sync,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
