# Kavach v2.0 — bank-agnostic, federated, threat-resilient fraud intelligence layer

Open-source (MIT). Implementation of the [v2 build brief](kavach-v2-build-brief.md): four modules
on top of the v1 single-bank pipeline (webhook → LLM classification → 3-tier rule response), plus
hardening against AI-specific attack surfaces.

> **Status: research prototype.** Every integration, regulatory and detection claim in this repo is
> labelled with what it has and has not been validated against. Read the *Limitations* section of
> each report before quoting any number.

## Layout

| Path | Component | What it is |
|---|---|---|
| `adapters/` | **A. Bank Adapter Registry** | Canonical `KavachTransaction` (Pydantic v2), fixed `FEATURE_CONTRACT`, abstract `BankAdapter`, `AdapterRegistry.ingest()`, per-bank `SecretProvider`, three *illustrative, unverified* adapters (Finacle / Temenos T24 / TCS BaNCS) |
| `core/` | v1 logic | `ThreatLevel` L0–L3 and the v1 rule classifier (**thresholds reconstructed** from the brief — see below) |
| `eval/` | **D. Evaluation Harness** | ULB dataset via OpenML (no Kaggle creds), adapter → schema → rules → per-level metrics → markdown report with limitations first; §5.2 adversarial suite |
| `federated/` | **B. Federated Threshold Learning** | Contract-agnostic online LR, FedAvg + coordinate-median + trimmed-mean aggregation, versioned model store, threshold bridge (fails to L2, never L0), `/feedback` contract, cold-start experiment |
| `hardening/` | **§5.1, §5.2** | Free-text sanitizer + data-block prompt renderer; 44-case adversarial corpus; voice-liveness interface + stub |
| `compliance/` | **C. RBI FR-2 generator** | Template generator with `<<MANUAL>>` placeholders, `filing_ready` always `False` until compliance clears the mapping |
| `tests/` | | 134 tests, all offline |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Reproducing the results

```bash
python -m eval.run_eval                              # ~15 s after first dataset fetch (~45 s, cached)
python -m federated.experiments.cold_start           # ~3–4 min
python -m eval.adversarial                           # <1 s
```

Outputs land in `eval/output/` and `federated/output/`.

## Headline results — read the caveats

### Component D — pipeline mechanics on ULB card-fraud data

| | AUC-PR | AUC-ROC |
|---|---|---|
| v1 rules (reconstructed; amount-only on this data) | 0.0018 | 0.535 |
| random ranker | 0.0016 | 0.5 |
| reference logistic regression on V1..V28 (not a Kavach component) | 0.760 | — |

**The rules are indistinguishable from random on this data, and that is the correct result.** The
dataset is European card-present fraud with anonymised PCA features. It has no phone-call context, no
sender age, no beneficiary-account age — none of Kavach's signals. Those fields were filled with
neutral constants so the rows could traverse the *real* adapter → schema → classifier path, which
means only amount rules can fire. The evaluation validates that the pipeline works end to end on
282,982 rows (1,825 zero-amount rows rejected by the schema and counted, 27 of them fraud); it says
nothing about UPI/IMPS social-engineering detection. Full report: [`eval/output/eval_report.md`](eval/output/eval_report.md).

### Component B — federated cold start

AUC-PR on a held-out "bank", mean over 15 cells (5 banks × 3 seeds), after identical fine-tuning on
*n* rows of the held-out bank's own labelled history:

| split | n | local-only | FedAvg | coord-median | pooled (upper bound) |
|---|---|---|---|---|---|
| `iid` (control) | 0 | 0.002 | **0.718** | 0.717 | 0.740 |
| `amount` (non-IID) | 0 | 0.002 | **0.671** | 0.664 | 0.699 |
| `cluster` (non-IID) | 0 | 0.003 | 0.593 | **0.616** | 0.658 |
| `amount` | 5000 | 0.581 | 0.679 | 0.669 | 0.721 |
| `cluster` | 5000 | 0.579 | 0.659 | 0.668 | 0.705 |

Federated beats local-only in **15/15 cells at cold start under every split**, capturing 90–97% of
the pooled-vs-local gap without any raw data leaving a partition. The win survives non-IID
partitioning (it was not an artefact of IID sampling). Under the `cluster` split the coordinate
median generalises *better* than FedAvg (13/15 cells) — the robust rule is not only for adversaries.

**Poisoning (§5.4)**, 4 contributors, 0 history on the held-out bank:

| attackers | attack | FedAvg | coord-median |
|---|---|---|---|
| 0 | — | 0.680 | 0.681 |
| 1 | inverted ×10, count ×20 | **0.001** | **0.689** |
| 1 | garbage N(0,5²), count ×20 | 0.003 | 0.686 |
| 2 of 4 | inverted | 0.001 | 0.001 |
| 2 of 4 | garbage | 0.012 | 0.346 |

One poisoned participant destroys FedAvg and does not move the median. Two of four is the median's
breakdown point and it fails there, as theory says — reported because it is the honest edge of the
guarantee. Full report: [`federated/output/cold_start_report.md`](federated/output/cold_start_report.md).

**Caveats:** synthetic partitions of one card-fraud dataset, not real banks; the ULB feature spec
(`ulb_creditcard_eval@v1[29]`) is evaluation-only — the production `kavach_transaction@v1[17]`
contract is exercised by a separate round-trip test, not by this experiment; attackers are crude;
no differential privacy or secure aggregation.

### §5.2 — prompt injection through free text

44/44 adversarial cases handled (27 targeting `remarks`, the UPI payer-typed field), 11 benign
Hindi/Hinglish/English controls pass unflagged. Structural attacks — delimiter forgery, role tokens,
override phrasing, zero-width/fullwidth/spacing/base64 obfuscation, JSON escape — are redacted, and
the system prompt is byte-identical in every case. **Not measured:** whether plain-language persuasion
that survives sanitization sways a live LLM. That needs a model and a domain red-team.

## Things that are explicitly not verified

1. **Adapter field mappings** (`adapters/*_adapter.py`) are illustrative, modelled on publicly visible
   conventions. No Finacle, Temenos or BaNCS API document was consulted. Each real bank needs its
   own mapping written against its actual contract.
2. **v1 rule thresholds** (`core/rules.py::RuleConfig`) are reconstructed from the brief; the n8n
   Switch-node export was not available. Drop the real values in and re-run `eval.run_eval`.
3. **RBI FR-2 structure** (`compliance/`) is a placeholder template. Every field is `verified=False`,
   every output is `filing_ready: false`, and the fraud category is never auto-filled.
4. **Voice liveness** (`hardening/voice_liveness.py`) is an interface and a stub. No detector exists
   here and none is claimed.
5. **The LLM node itself** was not exercised anywhere in this repo.

## Roadmap items deliberately out of scope

- Synthetic-identity / AI-generated KYC detection (§5.3) — an onboarding-time system, not a
  transaction-time one.
- Differential privacy / secure aggregation on federated deltas.
- A domain-appropriate labelled UPI/IMPS social-engineering dataset — the single thing that would
  turn the evaluation harness from a mechanics check into evidence.
