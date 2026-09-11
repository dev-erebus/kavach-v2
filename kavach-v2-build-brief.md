# Kavach v2.0 — Build Brief
### Bank-Agnostic, Federated, Threat-Resilient Fraud Intelligence Layer

**Purpose of this document:** This is an implementation specification, written to be handed to an engineering session (human or AI) as a precise build brief. It defines four new modules that convert Kavach from a single-bank demo into a system that can plausibly integrate across Indian banks, plus a hardening section for AI-specific attack surfaces. Personal application-essay content is intentionally excluded — that section only points at how to *use* this work honestly, not what to write.

---

## 0. What Changes From v1

v1 is a working single-bank demo: webhook → LLM classification → 3-tier rule response. v2 adds four modules on top of that core, without changing the underlying n8n/vLLM/Streamlit architecture:

| Module | Solves |
|---|---|
| A. Bank Adapter Registry | Every bank's core banking system speaks a different schema |
| B. Federated Threshold Learning | Hardcoded thresholds don't generalize across customer bases, and banks can't legally pool raw data |
| C. RBI FR-2 Auto-Report Generator | Level 3 incidents need a regulator-facing report, not just an internal dossier |
| D. Evaluation Harness | "AI-powered" is a claim; precision/recall on real data is evidence |

Plus a fifth section on hardening against AI-driven attack vectors that don't exist in v1 at all yet.

---

## 1. Component A — Bank Adapter Registry

### 1.1 The problem
SBI and PNB run Finacle. HDFC and Axis run Temenos T24. Some private banks run BaNCS. Each emits webhooks with different field names, nesting, date formats, and auth schemes. Kavach's canonical schema (`transaction_id`, `bank_id`, `amount_inr`, `sender_age`, `receiver_account_age_days`, `is_active_phone_call`, `device_location`) cannot assume any of these arrive pre-normalized.

### 1.2 What to build

```
/adapters
  ├── base_adapter.py       # Abstract BankAdapter class
  ├── registry.py           # bank_id → adapter instance resolution
  ├── schema.py             # Pydantic KavachTransaction canonical model
  ├── finacle_adapter.py    # Concrete adapter
  ├── temenos_adapter.py    # Concrete adapter
  └── bancs_adapter.py      # Concrete adapter
```

**`base_adapter.py`** — abstract class with two required methods:
- `normalize(raw_payload: dict) -> KavachTransaction` — maps bank-specific fields to canonical schema
- `validate_auth(headers: dict) -> bool` — verifies the bank's signing scheme (HMAC, mTLS cert, API key — varies per bank)

**`registry.py`** — a dict-based lookup resolved at webhook ingestion, before the payload reaches Node 2 of the n8n pipeline: `bank_id` in the incoming request header determines which adapter's `normalize()` runs first.

**`schema.py`** — the canonical `KavachTransaction` Pydantic model with field validators (e.g., `amount_inr > 0`, `sender_age` between 18–120).

**Important caveat to build in:** the field mappings in the three example adapters are illustrative, based on publicly documented core-banking schema conventions — not verified against real Finacle/Temenos/BaNCS API contracts. Each real integration requires the bank's actual API documentation before going live. Flag this explicitly in code comments so it's never mistaken for a validated integration.

### 1.3 Design principle
Onboarding a new bank should mean writing **one new adapter file plus one registry line** — never touching the core pipeline. This is what "integrable with every bank" actually requires architecturally.

### 1.4 Acceptance criteria
- New bank onboarding requires zero changes to Nodes 3–10 of the n8n pipeline
- 100% of canonical fields validated before reaching the LLM prompt layer
- Malformed or unmappable payloads are rejected with a structured error — never silently passed through with missing fields

---

## 2. Component B — Federated Per-Bank Threshold Learning

### 2.1 The problem
A ₹1,00,000 transfer is routine for one bank's urban salaried customer base and highly anomalous for another's rural pension-account base. Hardcoded thresholds can't reflect that. Separately, banks cannot legally pool raw transaction data with each other or a third party — data residency rules and the DPDP Act 2023 make that a non-starter.

### 2.2 Technical approach
- Each bank runs a **local, lightweight online classifier** (online logistic regression is enough for MVP — no need for anything exotic) trained incrementally on its own labeled fraud/non-fraud outcomes
- Periodically, only **model weight deltas** — never raw transaction data — are sent to a central aggregator
- The aggregator performs **federated averaging (FedAvg)** to produce a shared baseline model
- Each bank's live model = global baseline + local fine-tuning, so a brand-new bank with little history still starts from a reasonable baseline instead of zero

### 2.3 What to build

```
/federated
  ├── local_trainer.py       # Per-bank online learner
  ├── aggregator.py          # FedAvg implementation, scheduled job
  ├── model_store.py         # Versioned per-bank weight storage
  └── threshold_bridge.py    # Converts model probability → L0-L3 levels
```

`threshold_bridge.py` is the piece that replaces the hardcoded rule branches currently in n8n's Switch node — it takes the local model's output probability and maps it to a threat level using a per-bank calibrated cutoff instead of a fixed number.

A `/feedback` endpoint needs to exist so banks can report ground truth back ("this Level 2 freeze was a confirmed scam" / "this was a false positive") — that's what `local_trainer.py` incrementally learns from.

### 2.4 Acceptance criteria
- No raw transaction-level data ever leaves a bank's local environment — only weight deltas
- On a synthetic multi-bank split of a public dataset, the federated model should outperform a single bank's local-only model on a *held-out* bank's data (this is the cold-start test — it's the actual claim being made, so it needs to be demonstrated, not asserted)

---

## 3. Component C — RBI FR-2 Auto-Report Generator

### 3.1 The problem
A Level 3 hard-lock event isn't just an internal incident — RBI's cyber/fraud incident reporting framework requires structured disclosure. Right now Kavach generates a human-readable forensic dossier but nothing regulator-facing.

### 3.2 What to build

```
/compliance
  ├── fr2_generator.py    # Dossier + metadata → structured regulatory report
  └── field_mapping.py    # Canonical Kavach fields → RBI report fields
```

Output both a machine-readable JSON (for future API submission where a bank's compliance system supports it) and a human-readable PDF/markdown version for manual filing.

### 3.3 Critical caveat — do not skip this
**The exact RBI FR-2 field structure must be verified against the current RBI Master Direction and the integrating bank's own compliance/legal team before this touches production.** Regulatory formats change, and getting this wrong in a fraud-reporting context has real consequences. Build the module as a *template generator with clearly labeled placeholder fields*, and put an explicit `# COMPLIANCE REVIEW REQUIRED BEFORE PRODUCTION USE` comment at the top of the file. This isn't a formality — it's the honest way to build a regulatory feature you haven't had legally verified.

---

## 4. Component D — Evaluation Harness

### 4.1 Why this is the highest-leverage piece to build first
"AI-powered fraud detection" is a claim. "94% precision / 88% recall at Level 2+3 combined, evaluated on N thousand labeled transactions, with a documented false-positive breakdown" is evidence. This is also the single most useful thing for both real deployment credibility and any technical writeup — build it before, or in parallel with, everything else.

### 4.2 What to build

```
/eval
  ├── dataset_loader.py   # Loads IEEE-CIS or ULB/Kaggle Credit Card Fraud dataset
  ├── run_eval.py         # Runs full classification pipeline against labeled data
  ├── metrics.py          # Precision, recall, F1, AUC-PR, confusion matrix per level
  └── report.py           # Generates a markdown eval report with plots
```

### 4.3 Important honesty note — build this limitation into the report itself
Public fraud datasets (IEEE-CIS, ULB) are card-present/e-commerce fraud, not UPI/IMPS social-engineering fraud. They don't have a phone-call-context signal or elderly-targeting features. Using them is still valuable for validating the *classification pipeline mechanics*, but the eval report must explicitly state this gap rather than imply the numbers transfer directly to real Indian banking fraud patterns. An evaluation that acknowledges its own limitations is more credible than one that doesn't — reviewers notice the difference immediately.

---

## 5. Hardening Against Emerging AI-Driven Threats

This section is what "maximize future impact" actually means in practice — not a vague aspiration, but specific attack surfaces that exist *because* Kavach itself uses AI, plus attack patterns that are already emerging in real fraud.

### 5.1 AI voice cloning in social-engineering calls
`is_active_phone_call` currently only detects that a call is happening — it says nothing about whether the caller is who they claim to be. Voice cloning is already used in real scams targeting elderly victims, which is precisely Kavach's Level 2/3 population.
- **Build:** an optional integration point (interface only, not an in-house detector — that requires real audio ML expertise Kavach shouldn't fabricate) for third-party voice-liveness/deepfake-detection APIs
- **Add field:** `caller_verified_biometric: bool | None` — when unavailable, treat as *unknown*, never coerce to `False` or `True`

### 5.2 Prompt injection via transaction metadata
`device_location` and other free-text fields get interpolated into the LLM prompt at Node 3. If any of these fields are attacker-influenced, a crafted string could attempt to inject instructions that manipulate the classification itself.
- **Build:** strict sanitization on every free-text field before prompt interpolation — reject or escape instruction-like patterns
- **Build:** dedicated adversarial test cases in the eval harness specifically probing this
- **Principle:** no transaction metadata field should ever be able to alter system-prompt-level behavior, only user-prompt-level data

### 5.3 Synthetic identity / AI-generated KYC documents
Mule accounts increasingly pass onboarding KYC using AI-generated photos or documents. This is a different system boundary than Kavach's transaction-time layer — flag it as a **roadmap item for a complementary onboarding-time system**, not something to build into this pipeline. Scope discipline matters more than scope maximalism here.

### 5.4 Model poisoning in the federated loop
If the `/feedback` endpoint from Component B is gamed (an attacker submits false "confirmed fraud" or "false positive" labels), poisoned labels could corrupt the federated average across every bank.
- **Build:** outlier rejection on incoming weight deltas before aggregation — coordinate-median aggregation instead of naive averaging is a well-established, implementable defense against exactly this

### 5.5 The general principle underlying all four
Every new AI-threat mitigation should **degrade to "flag for human review," never to silent trust of an unverifiable signal.** That's both the correct safety posture and, separately, the kind of design reasoning that reads as mature engineering rather than feature-checklisting.

---

## 6. Suggested Build Order

1. Schema + Adapter Registry — foundational, unblocks everything else
2. Evaluation harness on the existing v1 rule-based logic — get honest baseline numbers before changing anything
3. Federated threshold learning — compare against the baseline from step 2
4. Prompt-injection hardening — do this before any real external input touches the LLM
5. RBI FR-2 generator — compliance layer, can be built in parallel with 3–4
6. Voice-liveness integration point — interface + stub only

---

## 7. Using This Work Honestly (Including for Applications)

A working system with real evaluation numbers and a stated limitations section is more useful — for a research application, a bank pilot conversation, or your own understanding of the system — than polished claims without evidence. If you're writing this up as a short technical note (problem, approach, results, limitations, future work), that structure is close to how early-stage research is actually communicated, and it's something you can defend in detail because you built and evaluated it yourself. That's the part no document can do for you, and it's also the part that actually matters most.

---

*Kavach v2.0 Build Brief · Implementation spec only · Regulatory and integration claims require verification before production use*
