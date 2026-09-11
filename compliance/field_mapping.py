# COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE
"""
Canonical Kavach fields → regulatory report fields.

==============================================================================
  THIS MAPPING IS A TEMPLATE. The target field names, groupings and
  enumerations below are PLACEHOLDERS shaped after the general structure of
  RBI fraud-reporting returns (bank identification / incident classification /
  amount / modus operandi / customer impact / action taken / timeline). They
  have NOT been verified against the current RBI Master Direction on Fraud Risk
  Management or against any bank's own FRMS/compliance schema. Every
  ``ReportField`` carries ``verified=False`` until a compliance reviewer sets it
  otherwise, and the generator refuses to mark output "ready for filing" while
  any field remains unverified.
==============================================================================

Two kinds of field
------------------
* ``AUTO`` — derived from a ``KavachTransaction`` + incident metadata. Filled in.
* ``MANUAL`` — needs a human (fraud officer / compliance) — e.g. the bank's
  official reporting officer, the RBI-assigned bank code, the fraud
  classification the bank's committee decides on. Emitted as a clearly labelled
  ``<<MANUAL: ...>>`` placeholder, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class Source(str, Enum):
    AUTO = "auto"  # derived from transaction / incident data
    MANUAL = "manual"  # must be entered by a human before filing


@dataclass(frozen=True)
class ReportField:
    key: str  # placeholder report key — TO BE REPLACED with the official field name
    section: str
    source: Source
    description: str
    #: Set by compliance review once the key/description match the official form. Never by code.
    verified: bool = False
    #: For AUTO fields: name of the extractor in fr2_generator.EXTRACTORS.
    extractor: Optional[str] = None


# Section names are descriptive placeholders, not the official part/annex numbering.
FIELDS: tuple[ReportField, ...] = (
    # ---- Part A: reporting entity ------------------------------------------
    ReportField("reporting_bank_name", "A. Reporting entity", Source.MANUAL, "Legal name of the reporting bank"),
    ReportField("reporting_bank_code", "A. Reporting entity", Source.MANUAL, "Bank code as assigned by the regulator (e.g. in the returns system)"),
    ReportField("reporting_officer", "A. Reporting entity", Source.MANUAL, "Name and designation of the authorised reporting officer"),
    ReportField("reporting_officer_contact", "A. Reporting entity", Source.MANUAL, "Official email / phone for regulator follow-up"),
    ReportField("branch_or_channel", "A. Reporting entity", Source.AUTO, "Channel / rail through which the transaction was initiated", extractor="rail"),
    # ---- Part B: incident identification -----------------------------------
    ReportField("internal_incident_id", "B. Incident identification", Source.AUTO, "Kavach incident reference", extractor="incident_id"),
    ReportField("bank_transaction_reference", "B. Incident identification", Source.AUTO, "Bank's own transaction id", extractor="transaction_id"),
    ReportField("date_of_occurrence", "B. Incident identification", Source.AUTO, "Transaction initiation date/time (IST)", extractor="occurred_at_ist"),
    ReportField("date_of_detection", "B. Incident identification", Source.AUTO, "When Kavach raised the Level 3 event (IST)", extractor="detected_at_ist"),
    ReportField("date_of_reporting", "B. Incident identification", Source.AUTO, "Report generation date (IST)", extractor="generated_at_ist"),
    ReportField("detection_to_report_hours", "B. Incident identification", Source.AUTO, "Elapsed hours detection → report; regulators impose deadlines here", extractor="detection_to_report_hours"),
    # ---- Part C: classification --------------------------------------------
    ReportField("fraud_category", "C. Classification", Source.MANUAL, "Regulatory fraud category — decided by the bank's fraud committee, not by Kavach"),
    ReportField("suspected_modus_operandi", "C. Classification", Source.AUTO, "Kavach's signal-based description of the suspected pattern (advisory)", extractor="modus_operandi"),
    ReportField("kavach_threat_level", "C. Classification", Source.AUTO, "Kavach level and the rule/model that produced it", extractor="threat_level"),
    ReportField("is_attempted_or_completed", "C. Classification", Source.AUTO, "Whether funds moved before the lock", extractor="attempt_status"),
    # ---- Part D: amount -----------------------------------------------------
    ReportField("amount_involved_inr", "D. Amount", Source.AUTO, "Transaction amount", extractor="amount_inr"),
    ReportField("amount_recovered_inr", "D. Amount", Source.MANUAL, "Recovered/blocked amount — known only after recovery action"),
    ReportField("amount_lost_inr", "D. Amount", Source.MANUAL, "Net loss after recovery — known only after recovery action"),
    # ---- Part E: parties ----------------------------------------------------
    ReportField("victim_customer_age_band", "E. Parties", Source.AUTO, "Sender age band (not exact age — data minimisation)", extractor="sender_age_band"),
    ReportField("victim_account_vintage", "E. Parties", Source.AUTO, "Sender account age band", extractor="sender_account_vintage"),
    ReportField("beneficiary_account_vintage", "E. Parties", Source.AUTO, "Beneficiary account age — mule indicator", extractor="receiver_account_vintage"),
    ReportField("beneficiary_bank", "E. Parties", Source.MANUAL, "Beneficiary bank and account details, from the bank's payment records"),
    ReportField("staff_involvement_suspected", "E. Parties", Source.MANUAL, "Whether bank staff involvement is suspected — human judgement"),
    # ---- Part F: indicators (the Kavach-specific value) --------------------
    ReportField("social_engineering_indicators", "F. Indicators", Source.AUTO, "Active call, voice-liveness verdict, new beneficiary, tampering flags", extractor="indicators"),
    # ---- Part G: action taken ----------------------------------------------
    ReportField("action_taken_by_bank", "G. Action taken", Source.AUTO, "Kavach's automated action (hard lock) — bank's subsequent actions are MANUAL", extractor="automated_action"),
    ReportField("subsequent_actions", "G. Action taken", Source.MANUAL, "Customer contact, account freeze extension, recovery steps, LEA referral"),
    ReportField("police_complaint_filed", "G. Action taken", Source.MANUAL, "FIR / cybercrime portal reference if filed"),
    # ---- Part H: attestation -----------------------------------------------
    ReportField("attestation", "H. Attestation", Source.MANUAL, "Signed attestation by the authorised officer that the above is accurate"),
)


def by_section() -> dict[str, list[ReportField]]:
    out: dict[str, list[ReportField]] = {}
    for f in FIELDS:
        out.setdefault(f.section, []).append(f)
    return out


def unverified() -> list[ReportField]:
    return [f for f in FIELDS if not f.verified]
