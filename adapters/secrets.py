"""
Per-bank secret lookup.

Secrets are **per bank, never per vendor**. SBI and PNB both run Finacle, but
they must never share an HMAC key — so the ``FinacleAdapter`` class cannot own a
secret; each *instance* (one per bank) asks a ``SecretProvider`` for that bank's
material at verification time.

Adapters never read environment variables or config files themselves. They get a
provider injected. This keeps auth logic unit-testable (``StaticSecretProvider``)
and keeps "where do secrets live" (vault, KMS, k8s secret, env) out of
integration code.

A provider returning ``None`` means "not configured" and every adapter treats
that as **fail closed**: accept nothing.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Protocol


class SecretProvider(Protocol):
    def get(self, bank_id: str, name: str) -> Optional[bytes]:
        """Return secret material for ``(bank_id, name)`` or ``None`` if unset.

        ``name`` is adapter-defined, e.g. ``"hmac_secret"``, ``"api_key"``,
        ``"mtls_cert_fingerprint"``. Values are bytes; callers decode if needed.
        """
        ...


class StaticSecretProvider:
    """In-memory provider for tests and local demos. Keys are ``(bank_id, name)``."""

    def __init__(self, secrets: Mapping[tuple[str, str], bytes | str]):
        self._s = {
            (b.lower(), n): (v.encode() if isinstance(v, str) else v) for (b, n), v in secrets.items()
        }

    def get(self, bank_id: str, name: str) -> Optional[bytes]:
        v = self._s.get((bank_id.lower(), name))
        return v or None


class EnvSecretProvider:
    """Reads ``KAVACH_SECRET__{BANK_ID}__{NAME}`` from the environment.

    Example: ``KAVACH_SECRET__SBI__HMAC_SECRET``. Intended for docker-compose /
    dev deployments; production should wrap a real vault behind the same
    ``SecretProvider`` protocol.
    """

    PREFIX = "KAVACH_SECRET__"

    def __init__(self, env: Optional[Mapping[str, str]] = None):
        self._env = env if env is not None else os.environ

    def get(self, bank_id: str, name: str) -> Optional[bytes]:
        key = f"{self.PREFIX}{bank_id.upper()}__{name.upper()}"
        v = self._env.get(key)
        return v.encode() if v else None
