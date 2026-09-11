# COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE
"""
Regulatory incident-report TEMPLATE generator for Level 3 events.

==============================================================================
  READ BEFORE USING THIS MODULE FOR ANYTHING OTHER THAN A DEMO

  This module produces a *template* — a structured draft with every field
  either auto-filled from Kavach data or marked ``<<MANUAL: ...>>``. It does
  NOT produce a filing-ready RBI return. Specifically:

  * The field set and layout in ``field_mapping.py`` are placeholders modelled
    on the general shape of fraud-reporting returns. They have not been
    checked against the current RBI Master Direction on Fraud Risk Management
    in Commercial Banks, the CPFIR/FMR return specifications, or the
    integrating bank's own compliance schema. Formats change; the version in
    force at filing time must be confirmed by the bank's compliance/legal team.
  * Kavach classifies *risk*. Whether an event is a reportable *fraud*, and in
    which regulatory category, is a determination the bank's fraud committee
    makes. The generator never fills those fields.
  * Every output carries ``filing_ready: false`` and an explicit list of the
    unverified fields and manual placeholders. ``FR2Report.assert_filing_ready``
    raises unless a compliance reviewer has cleared every field in the mapping.

  The honest value here is structural: when a Level 3 fires, the ops team gets
  a draft with the Kavach evidence already in the right places, a timeline
  with the detection→report clock running, and an unambiguous list of what a
  human must still supply. That saves time without pretending to be counsel.
==============================================================================

Outputs: ``to_json()`` (machine-readable, for a future API submission path once a
bank's compliance system supports it) and ``to_markdown()`` (human-readable, for
manual filing / attaching to the case file).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from adapters.schema import KavachTransaction
from core.levels import ThreatLevel

from .field_mapping import FIELDS, ReportField, Source, by_section, unverified

IST = ZoneInfo("Asia/Kolkata")
TEMPLATE_VERSION = "0.1-template-unverified"


@dataclass(frozen=True)
class IncidentMeta:
    """What the pipeline knows about the Level 3 event beyond the transaction itself."""

    incident_id: str
    detected_at_utc: datetime
    level: ThreatLevel
    decided_by: str  # e.g. "v1_rules:L3_high_amount_call_and_profile" or "threshold_bridge:p=0.987"
    funds_moved: Optional[bool]  # None = unknown at report time
    automated_action: str = "hard_lock_and_review"
    tampering_flags: dict[str, list[str]] = field(default_factory=dict)  # from §5.2 sanitizer
    dossier_reference: Optional[str] = None  # link/id of the human-readable forensic dossier


def _band(v: Optional[int], edges: tuple[int, ...], unit: str) -> str:
    if v is None:
        return "unknown"
    lo = 0
    for e in edges:
        if v < e:
            return f"{lo}–{e - 1} {unit}"
        lo = e
    return f"{lo}+ {unit}"


# --------------------------------------------------------------------------- #
# Extractors: AUTO field key → value. Each is deliberately tiny and data-minimising.
# --------------------------------------------------------------------------- #
EXTRACTORS: dict[str, Callable[[KavachTransaction, IncidentMeta, datetime], Any]] = {
    "rail": lambda tx, m, now: tx.rail.value,
    "incident_id": lambda tx, m, now: m.incident_id,
    "transaction_id": lambda tx, m, now: tx.transaction_id,
    "occurred_at_ist": lambda tx, m, now: tx.timestamp_utc.astimezone(IST).isoformat(timespec="seconds"),
    "detected_at_ist": lambda tx, m, now: m.detected_at_utc.astimezone(IST).isoformat(timespec="seconds"),
    "generated_at_ist": lambda tx, m, now: now.astimezone(IST).isoformat(timespec="seconds"),
    "detection_to_report_hours": lambda tx, m, now: round((now - m.detected_at_utc).total_seconds() / 3600, 2),
    "modus_operandi": lambda tx, m, now: _modus_operandi(tx, m),
    "threat_level": lambda tx, m, now: f"{m.level.name} ({m.level.action}); decided by {m.decided_by}",
    "attempt_status": lambda tx, m, now: {True: "completed — funds moved before lock", False: "attempted — blocked before settlement", None: "unknown at report time"}[m.funds_moved],
    "amount_inr": lambda tx, m, now: str(tx.amount_inr),
    "sender_age_band": lambda tx, m, now: _band(tx.sender_age, (30, 45, 60, 75), "years"),
    "sender_account_vintage": lambda tx, m, now: _band(tx.sender_account_age_days, (30, 180, 365, 1825), "days"),
    "receiver_account_vintage": lambda tx, m, now: _band(tx.receiver_account_age_days, (7, 30, 90, 365), "days"),
    "indicators": lambda tx, m, now: _indicators(tx, m),
    "automated_action": lambda tx, m, now: m.automated_action,
}


def _modus_operandi(tx: KavachTransaction, m: IncidentMeta) -> str:
    parts = []
    if tx.is_active_phone_call:
        parts.append("transfer initiated while sender was on an active call (social-engineering pattern)")
    if tx.caller_verified_biometric is False:
        parts.append("voice-liveness check FAILED on the active call (possible synthetic/cloned voice)")
    if tx.receiver_account_age_days <= 30:
        parts.append(f"beneficiary account only {tx.receiver_account_age_days} days old (mule indicator)")
    if tx.is_new_beneficiary:
        parts.append("first-ever payment to this beneficiary")
    if tx.sender_age >= 60:
        parts.append("elderly sender")
    if m.tampering_flags:
        parts.append("free-text fields contained instruction-like content (prompt-injection attempt)")
    return "; ".join(parts) if parts else "no Kavach-recognised social-engineering pattern; escalated on amount/model score"


def _indicators(tx: KavachTransaction, m: IncidentMeta) -> dict[str, Any]:
    return {
        "active_phone_call": tx.is_active_phone_call,
        "voice_liveness": {True: "verified_human", False: "FAILED", None: "unknown/not available"}[tx.caller_verified_biometric],
        "new_beneficiary": tx.is_new_beneficiary if tx.is_new_beneficiary is not None else "unknown",
        "beneficiary_account_age_days": tx.receiver_account_age_days,
        "free_text_tampering_flags": m.tampering_flags or {},
    }


# --------------------------------------------------------------------------- #
@dataclass
class FR2Report:
    template_version: str
    generated_at_utc: str
    bank_id: str
    values: dict[str, Any]  # key → value or "<<MANUAL: ...>>"
    manual_fields: list[str]
    unverified_fields: list[str]
    filing_ready: bool  # always False until compliance clears the mapping
    dossier_reference: Optional[str]

    def assert_filing_ready(self) -> None:
        if not self.filing_ready:
            raise RuntimeError(
                "Report is a TEMPLATE, not filing-ready: "
                f"{len(self.unverified_fields)} field definitions unverified by compliance, "
                f"{len(self.manual_fields)} manual placeholders unfilled."
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "_notice": "TEMPLATE — COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE. Field names are placeholders.",
                "template_version": self.template_version,
                "generated_at_utc": self.generated_at_utc,
                "bank_id": self.bank_id,
                "filing_ready": self.filing_ready,
                "dossier_reference": self.dossier_reference,
                "fields": self.values,
                "manual_fields": self.manual_fields,
                "unverified_field_definitions": self.unverified_fields,
            },
            indent=2,
            default=str,
        )

    def to_markdown(self) -> str:
        md = [
            "# Regulatory Incident Report — DRAFT TEMPLATE",
            "",
            "> **COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE.** Field names and structure are placeholders",
            "> not verified against the current RBI Master Direction or this bank's compliance schema.",
            f"> `filing_ready: {self.filing_ready}` · {len(self.manual_fields)} manual fields outstanding ·",
            f"> {len(self.unverified_fields)} field definitions unverified · template {self.template_version}",
            "",
            f"**Bank:** `{self.bank_id}` · **Generated (UTC):** {self.generated_at_utc}"
            + (f" · **Forensic dossier:** {self.dossier_reference}" if self.dossier_reference else ""),
            "",
        ]
        fields_by_key = {f.key: f for f in FIELDS}
        for section, fs in by_section().items():
            md.append(f"## {section}\n")
            md.append("| Field (placeholder key) | Value | Source | Verified |\n|---|---|---|---|")
            for f in fs:
                v = self.values.get(f.key)
                if isinstance(v, dict):
                    v = "<br>".join(f"{k}: {val}" for k, val in v.items())
                md.append(f"| `{f.key}`<br><sub>{f.description}</sub> | {v} | {f.source.value} | {'yes' if f.verified else '**no**'} |")
            md.append("")
        md.append("## Outstanding before filing\n")
        for k in self.manual_fields:
            md.append(f"- [ ] `{k}` — {fields_by_key[k].description}")
        md.append("")
        md.append("_Kavach classifies risk. Whether this event is a reportable fraud, and its category, is the bank's determination._")
        return "\n".join(md)


def generate(tx: KavachTransaction, meta: IncidentMeta, now: Optional[datetime] = None) -> FR2Report:
    """Build the template. Only Level 3 events are eligible; anything else is a caller bug."""
    if meta.level is not ThreatLevel.L3:
        raise ValueError(f"regulatory report template is for L3 events only; got {meta.level.name}")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    values: dict[str, Any] = {}
    manual: list[str] = []
    for f in FIELDS:
        if f.source is Source.MANUAL:
            values[f.key] = f"<<MANUAL: {f.description}>>"
            manual.append(f.key)
        else:
            values[f.key] = EXTRACTORS[f.extractor](tx, meta, now)

    unv = [f.key for f in unverified()]
    return FR2Report(
        template_version=TEMPLATE_VERSION,
        generated_at_utc=now.isoformat(timespec="seconds"),
        bank_id=tx.bank_id,
        values=values,
        manual_fields=manual,
        unverified_fields=unv,
        filing_ready=(not unv and not manual),  # both must be empty; today neither ever is
        dossier_reference=meta.dossier_reference,
    )
