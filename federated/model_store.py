"""
Versioned weight storage — per bank and for the global model.

Layout on disk (JSON, one file per version, human-readable on purpose):

    <root>/<spec_key_slug>/global/v0003.json
    <root>/<spec_key_slug>/banks/sbi/v0007.json

Every record carries the ``spec_key`` it was trained under and the store
refuses to load a record under a different spec. This is the last line of the
fixed-feature-contract defence: even if a mis-versioned model file is copied
into the wrong directory, it cannot be served.

What is stored: parameters, spec key, version, timestamp, sample count, the
global version it was fine-tuned from, and free-form ``meta``. What is never
stored here: any transaction, feature vector or label.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .feature_spec import REGISTRY, FeatureSpec, SpecMismatch


@dataclass
class ModelRecord:
    spec_key: str
    scope: str  # "global" or a bank_id
    version: int
    params: list[float]
    created_utc: str
    n_samples: int = 0
    base_global_version: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def params_array(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


class ModelStore:
    def __init__(self, root: Path, spec: FeatureSpec):
        self.root = Path(root)
        self.spec = spec
        self._base = self.root / _slug(spec.key)
        (self._base / "global").mkdir(parents=True, exist_ok=True)
        (self._base / "banks").mkdir(parents=True, exist_ok=True)

    # ---- paths -------------------------------------------------------------
    def _dir(self, scope: str) -> Path:
        d = self._base / "global" if scope == "global" else self._base / "banks" / _slug(scope)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _versions(self, scope: str) -> list[int]:
        return sorted(int(p.stem[1:]) for p in self._dir(scope).glob("v*.json"))

    # ---- write -------------------------------------------------------------
    def save(
        self,
        scope: str,
        params: np.ndarray,
        n_samples: int = 0,
        base_global_version: Optional[int] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> ModelRecord:
        p = np.asarray(params, dtype=float)
        if p.shape != (self.spec.dim + 1,):
            raise SpecMismatch(f"params shape {p.shape} does not match {self.spec.key} (+bias)")
        if not np.all(np.isfinite(p)):
            raise SpecMismatch("refusing to store non-finite parameters")
        vs = self._versions(scope)
        version = (vs[-1] + 1) if vs else 1
        rec = ModelRecord(
            spec_key=self.spec.key,
            scope=scope,
            version=version,
            params=p.tolist(),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_samples=int(n_samples),
            base_global_version=base_global_version,
            meta=dict(meta or {}),
        )
        (self._dir(scope) / f"v{version:04d}.json").write_text(json.dumps(asdict(rec), indent=1), encoding="utf-8")
        return rec

    # ---- read --------------------------------------------------------------
    def load(self, scope: str, version: Optional[int] = None) -> ModelRecord:
        vs = self._versions(scope)
        if not vs:
            raise FileNotFoundError(f"no stored model for scope '{scope}' under {self.spec.key}")
        v = vs[-1] if version is None else version
        if v not in vs:
            raise FileNotFoundError(f"version {v} not found for scope '{scope}'")
        raw = json.loads((self._dir(scope) / f"v{v:04d}.json").read_text(encoding="utf-8"))
        rec = ModelRecord(**raw)
        if rec.spec_key != self.spec.key:
            raise SpecMismatch(f"stored model is {rec.spec_key}; store is serving {self.spec.key}")
        if len(rec.params) != self.spec.dim + 1:
            raise SpecMismatch(f"stored params have length {len(rec.params)}, expected {self.spec.dim + 1}")
        return rec

    def latest_version(self, scope: str) -> Optional[int]:
        vs = self._versions(scope)
        return vs[-1] if vs else None

    def scopes(self) -> list[str]:
        banks = sorted(p.name for p in (self._base / "banks").iterdir() if p.is_dir())
        return (["global"] if self._versions("global") else []) + banks

    @staticmethod
    def spec_for_key(key: str) -> FeatureSpec:
        try:
            return REGISTRY[key]
        except KeyError:
            raise SpecMismatch(f"unknown spec key '{key}'; registered: {sorted(REGISTRY)}")
