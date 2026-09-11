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

AUC-PR on a held-out "bank" (mean over 15 cells = 5 banks × 3 seeds), after identical fine-tuning on
*n* rows of the held-out bank's own labelled history. Because AUC-PR's floor is the prevalence and
held-out prevalence varies across banks under the non-IID splits (3.3× under `amount`, 11.8× under `cluster` in seed 0), **lift** (AUC-PR ÷ that
cell's own prevalence) is shown alongside; a constant scorer has lift exactly 1.

| split | n | local-only | FedAvg | coord-median | pooled (upper bound) | FedAvg beats local |
|---|---|---|---|---|---|---|
| `amount` (non-IID) | **50** | 0.016 (11×) | **0.672 (476×)** | 0.665 (469×) | 0.705 (502×) | 15/15 |
| `amount` | **500** | 0.326 (183×) | **0.676 (478×)** | 0.668 (470×) | 0.706 (503×) | 13/15 |
| `amount` | 5000 | 0.581 (372×) | 0.679 (475×) | 0.669 (466×) | 0.721 (508×) | 10/15 |
| `cluster` (non-IID) | **50** | 0.107 (38×) | 0.620 (498×) | **0.638 (518×)** | 0.682 (557×) | 14/15 |
| `cluster` | **500** | 0.316 (102×) | 0.652 (518×) | **0.666 (533×)** | 0.710 (573×) | 12/15 |
| `cluster` | 5000 | 0.579 (417×) | 0.659 (514×) | 0.668 (524×) | 0.705 (568×) | 14/15 |
| `iid` (control) | 50 | 0.081 (48×) | 0.696 (410×) | 0.697 (410×) | 0.725 (426×) | 15/15 |
| `iid` | 500 | 0.283 (168×) | 0.705 (414×) | 0.703 (413×) | 0.725 (426×) | 15/15 |
| *all splits* | *0 (degenerate)* | *≈prevalence (1×)* | *0.59–0.72* | *0.62–0.72* | *0.66–0.74* | *15/15* |

**The result is the n=50 and n=500 rows.** There the local-only model has real labelled data (on
average 0.1 and 0.9 fraud rows respectively) and is genuinely trying to rank; federation still beats
it by 0.3–0.65 AUC-PR and captures 89–96% of the pooled-vs-local gap without any raw data leaving a
partition. The win survives non-IID partitioning, and it holds on lift as well as on raw AUC-PR
(per-cell wins are identical by construction; the cross-bank means keep the same ordering). At n=0
local-only is a constant scorer — that row says only that a model beats no model and is not the headline.

Per-held-out-bank tables in the report attribute the spread: at n=500 the correlation between a cell's
AUC-PR and its own prevalence is r≈+0.8 under `cluster`, i.e. most of the raw sd there is base rate,
not method. Under `cluster` the coordinate median generalises *better* than FedAvg (12–13/15 cells) —
the robust rule is not only for adversaries.

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
guarantee.

**Threshold bridge — per-bank calibrated cutoffs vs one global cutoff (the brief's actual §2.3 claim).**
Same FedAvg model at every bank; only the cutoffs differ; ≥L2 budget 1% of each bank's traffic:

| split | per-bank L2 cutoffs differ by | global cutoff: alert volume range | per-bank: alert volume range | Δ pooled recall (per-bank − global) |
|---|---|---|---|---|
| `iid` | 1.1× | 0.94–1.10% | 0.85–1.19% | +0.0 pts |
| `amount` | 2.6× | **0.44–3.19%** | 0.87–1.18% | +0.1 pts |
| `cluster` | 3.7× | **0.36–2.56%** | 0.93–1.22% | −2.1 pts |

What the bridge demonstrably buys is **alert-budget adherence per bank** — a single cutoff sends 3% of
one bank's traffic to review and 0.4% of another's against the same 1% budget. What it does **not** buy
on this data is more fraud caught: for the same total alerts, pooled recall is equal or slightly *better*
under the global cutoff, because comparable probabilities plus one cutoff is already the recall-optimal
allocation of a fixed budget. Labelled per-bank *precision* calibration is unreliable at tens of frauds
per bank (target met in 21/45 cells vs 22/45 for the global cutoff, flipping by split) and should wait
for `/feedback` volume. Full report: [`federated/output/cold_start_report.md`](federated/output/cold_start_report.md).

**Caveats:** synthetic partitions of one card-fraud dataset, not real banks; the ULB feature spec
(`ulb_creditcard_eval@v1[29]`) is evaluation-only — the production `kavach_transaction@v1[17]`
contract is exercised by a separate round-trip test, not by this experiment; attackers are crude;
no differential privacy or secure aggregation; the bridge experiment uses the same global model at
every bank (no per-bank fine-tuning) to isolate the cutoff effect.

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
