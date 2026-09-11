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
| `tests/` | | 145 tests, all offline |

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Reproducing the results

```bash
python -m eval.run_eval                              # ~15 s after first dataset fetch (~45 s, cached)
python -m federated.experiments.cold_start           # ~4 min (cold start + poisoning + bridge)
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

AUC-PR on a held-out "bank" (mean over 15 cells = 5 banks × 3 seeds) after identical fine-tuning on *n*
rows of the held-out bank's own labelled history. Lift = AUC-PR ÷ that cell's own prevalence (a constant
scorer has lift 1). **A local model with no fraud row in its history cannot rank**, so the column
"cells with ≥1 fraud" says how many of the 15 comparisons were against a baseline that could do the task.

| split | n | cells with ≥1 fraud in history | local-only | FedAvg | coord-median | pooled (upper bound) | FedAvg beats local |
|---|---|---|---|---|---|---|---|
| `amount` (non-IID) | **5000** | 15/15 (min 2, mean 10.1) | 0.581 | **0.679** | 0.669 | 0.721 | **10/15** |
| `cluster` (non-IID) | **5000** | 15/15 (min 1, mean 13.1) | 0.579 | 0.659 | **0.668** | 0.705 | **14/15** |
| `iid` (control) | **5000** | 15/15 (min 6, mean 9.4) | 0.631 | **0.692** | 0.691 | 0.728 | **14/15** |
| `amount` | 500 | 8/15 | 0.326 | 0.676 | 0.668 | 0.706 | 13/15 |
| `cluster` | 500 | 7/15 | 0.316 | 0.652 | 0.666 | 0.710 | 12/15 |
| `amount` | 50 | 0/15 | 0.016 | 0.672 | 0.665 | 0.705 | 15/15 |
| `cluster` | 50 | 2/15 | 0.107 | 0.620 | 0.638 | 0.682 | 14/15 |
| *all* | *0* | *0/15* | *≈prevalence* | *0.59–0.72* | *0.62–0.72* | *0.66–0.74* | *15/15* |

**The headline is n=5000**, the only history size at which every cell's local model had fraud rows to
learn from. There the baseline is genuinely trying and federation still wins by +0.06 to +0.10 AUC-PR
(10/15, 14/15, 14/15 cells), closing 63–70% of the pooled-vs-local gap without raw data leaving a
partition. That is a smaller claim than the n=0 numbers suggest, and it is the defensible one.

**n=50 is near-degenerate and n=500 is mixed.** At n=50 only 0–2 of 15 cells had *any* fraud row in the
local history; at n=500, 7–9. Splitting n=500 by that condition: in cells with **no** fraud row local-only
sits at ≈0.01 and federation wins every cell — that is the n=0 comparison again. In cells **with** a fraud
row, local-only jumps to 0.598 (`amount`) / 0.664 (`cluster`) / 0.466 (`iid`) and federation's margin
shrinks to +0.095 (6/8) / +0.039 (4/7) / +0.230 (9/9).

**So what federation fixes here is positive-class scarcity, not sample scarcity.** One fraud label moves
the local model from ≈0.01 to ≈0.5–0.7 AUC-PR; the other 499 legitimate rows barely matter. A new bank
has plenty of transactions and almost no confirmed fraud labels — the federated baseline supplies the
ranking direction only positives can teach, from banks that have them.

Under `cluster` the coordinate median beats FedAvg with no attacker present (10–13/15 cells across history
sizes); under `amount` it costs ≤0.011. Per-held-out-bank tables and a Pearson *r* between cell AUC-PR
and cell prevalence (+0.82 under `cluster` at n=5000, ≈+0.5 elsewhere) are in the report.

**Poisoning (§5.4)** — 4 contributors, one or two submit inverted (−10×) or garbage deltas with a 20×
inflated sample count; the held-out bank then fine-tunes on 0 / 500 / 5000 of its own rows:

| attackers | attack | FedAvg n=0 | median n=0 | FedAvg n=5000 | median n=5000 |
|---|---|---|---|---|---|
| 0 | — | 0.682 | 0.682 | 0.688 | 0.680 |
| 1 | inverted | **0.001** | **0.691** | **0.001** | **0.680** |
| 1 | garbage | 0.003 | 0.689 | 0.010 | 0.681 |
| 2 of 4 | inverted | 0.001 | 0.001 | 0.001 | 0.001 |
| 2 of 4 | garbage | 0.012 | 0.349 | 0.039 | 0.441 |

n=0 is the attack-favourable case; n=5000 is the realistic one for a bank that has been live. **Local
fine-tuning recovers 0% of FedAvg's loss** under one inverted attacker even with 5000 rows (~9 fraud labels):
the poisoned weights sit too far from the origin for the same fine-tune budget to return them. The median is
the defence at every history size; two of four attackers is its breakdown point and it fails there, as theory
says. Design implication (§5.5): a bank should score an incoming global model on its own labelled history
before adopting it — detection, not recovery, is what local data buys.

**Threshold bridge — per-bank cutoffs vs one global cutoff (the brief's §2.3 claim).** Four rows hold one
FedAvg model fixed at every bank so only the cutoffs differ; a fifth (`fullstack`) adds per-bank fine-tuning
before per-bank calibration — the brief's full §2.2 stack. ≥L2 budget 1% of each bank's traffic:

| split | per-bank L2 cutoffs differ by | global cutoff: alert volume | per-bank cutoffs | full stack | pooled ≥L2 recall: global / per-bank / full stack |
|---|---|---|---|---|---|
| `amount` | 2.5× | **0.37–3.11%** | 0.86–1.14% | 0.79–1.23% | 86.1 / 85.7 / 85.3% |
| `cluster` | 3.8× | **0.33–2.64%** | 0.85–1.17% | 0.85–1.19% | 85.4 / 83.5 / 85.4% |
| `iid` | 1.1× | 0.89–1.19% | 0.82–1.26% | 0.84–1.26% | 85.6 / 85.6 / 84.2% |

What the bridge demonstrably buys is **alert-budget adherence per bank**: a single cutoff sends 3% of one
bank's traffic to review and 0.4% of another's against the same 1% budget. What it does **not** buy on this
data is more fraud caught — for the same total alerts, pooled recall is within 0.4 pts under `amount` and
1.9 pts *worse* under `cluster`, because comparable probabilities plus one cutoff is already the recall-optimal
allocation of a fixed budget. Adding per-bank fine-tuning (full stack) changes pooled recall by −1.4 to +1.9
pts — noise at this fraud volume. Labelled per-bank *precision* calibration meets its target in 25/45 cells vs
22/45 for the global cutoff, flipping by split; it should wait for `/feedback` volume. Full report:
[`federated/output/cold_start_report.md`](federated/output/cold_start_report.md).

**Caveats:** synthetic partitions of one card-fraud dataset, not real banks; the ULB feature spec
(`ulb_creditcard_eval@v1[29]`) is evaluation-only — the production `kavach_transaction@v1[17]` contract is
exercised by a separate round-trip test, not by this experiment; attackers are crude; no differential
privacy or secure aggregation; logistic regression throughout, so a stronger local learner would shrink the
n=5000 gap.

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
