"""
Bank adapter registry — resolves ``bank_id`` → adapter at webhook ingestion.

This runs *before* Node 2 of the n8n pipeline. Its job is to be the only place
in the system that knows more than one bank exists. Everything downstream gets a
``KavachTransaction`` and nothing else.

Onboarding a new bank (brief §1.3):

    1. write ``adapters/<vendor>_adapter.py`` (or reuse one — many banks share a vendor)
    2. add one ``registry.register(...)`` line

Nodes 3–10 never change.

Ingestion contract
------------------
``AdapterRegistry.ingest(headers, body)`` performs, in order:

    1. read ``X-Kavach-Bank-Id`` header            → ``NormalizationError(unknown_bank)`` if absent/unknown
    2. ``adapter.validate_auth(headers, body)``     → ``AuthenticationError`` if False
    3. parse JSON body with ``parse_float=Decimal`` → ``NormalizationError(bad_value)`` if not JSON
    4. ``adapter.normalize(payload)``               → ``NormalizationError`` on any mapping failure

Auth is checked *before* the body is parsed so an unauthenticated caller can't
probe the schema by sending malformed payloads and reading error messages.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Iterable, Mapping, Optional

from .base_adapter import (
    AuthenticationError,
    BankAdapter,
    NormalizationError,
    header_lookup,
)
from .schema import KavachTransaction

BANK_ID_HEADER = "X-Kavach-Bank-Id"


class AdapterRegistry:
    def __init__(self, adapters: Iterable[BankAdapter] = ()):
        self._adapters: dict[str, BankAdapter] = {}
        for a in adapters:
            self.register(a)

    # ---------------------------------------------------------------- setup
    def register(self, adapter: BankAdapter) -> None:
        key = adapter.bank_id.lower()
        if key in self._adapters:
            # Two adapters for one bank is always a config error; refuse loudly.
            raise ValueError(f"bank_id '{key}' already registered")
        self._adapters[key] = adapter

    def resolve(self, bank_id: Optional[str]) -> BankAdapter:
        if not bank_id or bank_id.lower() not in self._adapters:
            raise NormalizationError(
                bank_id=bank_id or "<missing>",
                code="unknown_bank",
                message=f"no adapter registered for bank_id '{bank_id}'",
                fields=[BANK_ID_HEADER],
            )
        return self._adapters[bank_id.lower()]

    @property
    def bank_ids(self) -> list[str]:
        return sorted(self._adapters)

    # ------------------------------------------------------------ ingestion
    def ingest(self, headers: Mapping[str, str], body: bytes) -> KavachTransaction:
        """Full webhook step: resolve → authenticate → parse → normalize."""
        adapter = self.resolve(header_lookup(headers, BANK_ID_HEADER))

        if not adapter.validate_auth(headers, body):
            raise AuthenticationError(f"[{adapter.bank_id}] webhook signature/credential rejected")

        try:
            # parse_float=Decimal: JSON numbers like 150000.00 must not become
            # binary floats before they reach the money validator.
            payload: Any = json.loads(body, parse_float=Decimal)
        except (ValueError, UnicodeDecodeError) as e:
            raise NormalizationError(
                bank_id=adapter.bank_id,
                code="bad_value",
                message=f"request body is not valid JSON: {e}",
                fields=["<body>"],
            ) from e
        if not isinstance(payload, dict):
            raise NormalizationError(
                bank_id=adapter.bank_id,
                code="bad_value",
                message="request body must be a JSON object",
                fields=["<body>"],
            )
        return adapter.normalize(payload)


def default_registry(secrets) -> AdapterRegistry:
    """The illustrative demo wiring. Every mapping here is UNVERIFIED (see each adapter).

    Bank→vendor assignments reflect widely reported public information about
    which core-banking product each bank runs, but Kavach has no relationship
    with any of these banks and none of this has been validated against their
    actual APIs.
    """
    from .bancs_adapter import BancsAdapter
    from .finacle_adapter import FinacleAdapter
    from .temenos_adapter import TemenosAdapter

    return AdapterRegistry(
        [
            FinacleAdapter("sbi", secrets),  # SBI: Finacle  (public reporting)
            FinacleAdapter("pnb", secrets),  # PNB: Finacle  (public reporting)
            TemenosAdapter("hdfc", secrets),  # HDFC: Temenos T24 (public reporting)
            TemenosAdapter("axis", secrets),  # Axis: Temenos T24 (public reporting)
            BancsAdapter("demo_bancs_bank", secrets),  # placeholder; TCS BaNCS user
        ]
    )
