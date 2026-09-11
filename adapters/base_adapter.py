"""
Abstract ``BankAdapter``.

An adapter is the *only* code in Kavach that knows anything about a specific
bank's core-banking system. Its contract is deliberately tiny:

* ``validate_auth(headers)``  — is this webhook really from the bank?
* ``normalize(raw_payload)``  — turn the bank's payload into a ``KavachTransaction``

Everything else in the pipeline consumes ``KavachTransaction`` and nothing else,
which is what makes "onboard a new bank = one adapter file + one registry line"
true rather than aspirational (brief §1.3).

Error contract (brief §1.4, acceptance criterion 3)
----------------------------------------------------
A malformed or unmappable payload must be rejected with a **structured** error,
never silently passed through with missing fields. Adapters raise
``NormalizationError``; it carries the ``bank_id``, a machine-readable
``code``, and the list of offending fields so the webhook layer can return a
useful 4xx and the ops team can see *which* mapping broke.

Concrete adapters should not catch and swallow anything — ``BankAdapter._build``
wraps Pydantic ``ValidationError`` into ``NormalizationError`` for them.
"""

from __future__ import annotations

import hmac
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .schema import KavachTransaction

IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class AdapterError(Exception):
    """Base for all adapter-layer failures."""


@dataclass
class NormalizationError(AdapterError):
    """Structured rejection of a payload that could not be mapped to the canonical schema.

    ``code`` values:
        ``missing_field``      — a required bank field was absent
        ``bad_value``          — a field was present but unparseable / out of range
        ``schema_violation``   — mapped dict failed ``KavachTransaction`` validation
        ``unknown_bank``       — no adapter registered for ``bank_id``
    """

    bank_id: str
    code: str
    message: str
    fields: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - trivial
        f = f" fields={self.fields}" if self.fields else ""
        return f"[{self.bank_id}] {self.code}: {self.message}{f}"

    def to_dict(self) -> dict[str, Any]:
        """Shape returned to the bank in the webhook 4xx body."""
        return {
            "error": "normalization_failed",
            "bank_id": self.bank_id,
            "code": self.code,
            "message": self.message,
            "fields": list(self.fields),
        }


class AuthenticationError(AdapterError):
    """Webhook failed the bank's signing / auth scheme."""


def header_lookup(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive HTTP header lookup."""
    lname = name.lower()
    for k, v in headers.items():
        if k.lower() == lname:
            return v
    return None


# --------------------------------------------------------------------------- #
# Abstract adapter
# --------------------------------------------------------------------------- #
class BankAdapter(ABC):
    """Contract every bank integration must satisfy.

    Subclasses set ``bank_id`` and implement the two abstract methods. Helper
    methods below cover the boring, error-prone parts (required-field lookup,
    nested-path access, IST→UTC conversion, HMAC verification) so concrete
    adapters stay short and hard to get wrong.
    """

    #: Kavach-side identifier; must match the key used in ``registry.py``.
    bank_id: str = "abstract"

    #: Human-readable core-banking product name, for logs / dossier.
    core_banking_system: str = "unknown"

    # ---- required interface ------------------------------------------------
    @abstractmethod
    def normalize(self, raw_payload: Mapping[str, Any]) -> KavachTransaction:
        """Map a bank-specific webhook payload to the canonical schema.

        Must raise ``NormalizationError`` on any unmappable input. Must never
        return a partially-populated record.
        """

    @abstractmethod
    def validate_auth(self, headers: Mapping[str, str], body: bytes = b"") -> bool:
        """Verify the webhook came from the bank (HMAC, API key, mTLS header, ...).

        Returns ``True``/``False``; does not raise. The webhook layer converts
        ``False`` into a 401 before ``normalize`` is ever called.
        """

    # ---- helpers for subclasses -------------------------------------------
    def _require(self, payload: Mapping[str, Any], *path: str) -> Any:
        """Fetch a (possibly nested) key; raise structured error if missing/None."""
        cur: Any = payload
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur or cur[key] is None:
                raise NormalizationError(
                    bank_id=self.bank_id,
                    code="missing_field",
                    message=f"required field '{'.'.join(path)}' absent from payload",
                    fields=[".".join(path)],
                )
            cur = cur[key]
        return cur

    def _optional(self, payload: Mapping[str, Any], *path: str, default: Any = None) -> Any:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur or cur[key] is None:
                return default
            cur = cur[key]
        return cur

    def _section(self, payload: Mapping[str, Any], *path: str) -> "Section":
        """Return a view onto a nested sub-object that reports full dotted paths in errors."""
        return Section(self, self._require(payload, *path), ".".join(path))

    def _parse_datetime(
        self, value: Any, fmt: Optional[str], assume_tz: ZoneInfo, field_name: str
    ) -> datetime:
        """Parse a bank timestamp; attach ``assume_tz`` if the string carries no offset."""
        try:
            if isinstance(value, (int, float)):
                # epoch seconds (or ms if it's obviously too large)
                ts = float(value)
                if ts > 1e11:
                    ts /= 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            if fmt:
                dt = datetime.strptime(str(value), fmt)
            else:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            raise NormalizationError(
                bank_id=self.bank_id,
                code="bad_value",
                message=f"could not parse datetime '{value}': {e}",
                fields=[field_name],
            ) from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=assume_tz)
        return dt.astimezone(timezone.utc)

    def _parse_bool(self, value: Any, field_name: str) -> bool:
        """Accept Y/N, true/false, 1/0 — reject anything ambiguous."""
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"y", "yes", "true", "1", "t"}:
            return True
        if s in {"n", "no", "false", "0", "f"}:
            return False
        raise NormalizationError(
            bank_id=self.bank_id,
            code="bad_value",
            message=f"could not interpret '{value}' as boolean",
            fields=[field_name],
        )

    def _parse_tristate(self, value: Any, field_name: str) -> Optional[bool]:
        """Like ``_parse_bool`` but ``None``/''/'UNKNOWN' → ``None`` (see §5.1)."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "unknown", "na", "n/a", "null"}:
            return None
        return self._parse_bool(value, field_name)

    def _build(self, mapped: dict[str, Any]) -> KavachTransaction:
        """Construct the canonical model, converting Pydantic errors to structured ones."""
        try:
            return KavachTransaction(**mapped)
        except ValidationError as e:
            bad = sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()})
            raise NormalizationError(
                bank_id=self.bank_id,
                code="schema_violation",
                message="; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
                ),
                fields=bad,
            ) from e

    @staticmethod
    def _hmac_matches(secret: bytes, body: bytes, presented_hex: str, digest=hashlib.sha256) -> bool:
        """Constant-time HMAC comparison. Used by adapters whose bank signs the body."""
        if not presented_hex:
            return False
        expected = hmac.new(secret, body, digest).hexdigest()
        return hmac.compare_digest(expected.lower(), presented_hex.strip().lower())


@dataclass
class Section:
    """A nested sub-dict of a payload, remembering its dotted prefix for error reporting."""

    adapter: BankAdapter
    data: Mapping[str, Any]
    prefix: str

    def path(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def require(self, key: str) -> Any:
        if not isinstance(self.data, Mapping) or key not in self.data or self.data[key] is None:
            raise NormalizationError(
                bank_id=self.adapter.bank_id,
                code="missing_field",
                message=f"required field '{self.path(key)}' absent from payload",
                fields=[self.path(key)],
            )
        return self.data[key]

    def optional(self, key: str, default: Any = None) -> Any:
        if not isinstance(self.data, Mapping):
            return default
        v = self.data.get(key)
        return default if v is None else v

    # typed conveniences -------------------------------------------------
    def require_int(self, key: str) -> int:
        return self._int(self.require(key), key)

    def optional_int(self, key: str) -> Optional[int]:
        v = self.optional(key)
        return None if v is None else self._int(v, key)

    def require_bool(self, key: str) -> bool:
        return self.adapter._parse_bool(self.require(key), self.path(key))

    def optional_tristate(self, key: str) -> Optional[bool]:
        return self.adapter._parse_tristate(self.optional(key), self.path(key))

    def require_datetime(self, key: str, fmt: Optional[str], assume_tz: ZoneInfo) -> datetime:
        return self.adapter._parse_datetime(self.require(key), fmt, assume_tz, self.path(key))

    def _int(self, value: Any, key: str) -> int:
        try:
            return int(str(value).strip())
        except (ValueError, TypeError) as e:
            raise NormalizationError(
                bank_id=self.adapter.bank_id,
                code="bad_value",
                message=f"expected integer, got '{value}'",
                fields=[self.path(key)],
            ) from e
