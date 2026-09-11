# Kavach v2 — Results

*Every number here appears in a table in `eval/output/eval_report.md`, `eval/output/adversarial_report.md`
or `federated/output/cold_start_report.md`. Nothing is asserted that is not measured there.*

## 1. Problem

Kavach v1 is a single-bank demo: a core-banking webhook is classified by an LLM and a fixed rule set into
four threat levels (L0 allow → L3 hard-lock), aimed at UPI/IMPS social-engineering fraud against elderly
customers. Three things stop it from generalising. Every bank's core system emits a different schema. A
threshold that is routine for one bank's customers is anomalous for another's, and banks cannot pool
transaction data to learn a shared one (DPDP Act 2023, data-residency rules). And because the system is
itself an LLM pipeline fed attacker-influenced text, it has attack surfaces v1 never considered.

v2 adds a bank-adapter layer with a fixed feature contract, federated per-bank threshold learning,
prompt-injection hardening, a regulatory-report template, and — first — an evaluation harness so that
claims about the pipeline are measured rather than asserted.

## 2. Approach

**Adapters.** A canonical Pydantic `KavachTransaction` with `extra="forbid"`, a `FEATURE_CONTRACT` fixing
a 17-slot feature vector (every optional field gets a value slot and a missing-indicator slot, transforms
fixed rather than fitted), an abstract `BankAdapter`, per-bank secrets via an injected provider, and three
illustrative adapters (Finacle / Temenos T24 / TCS BaNCS) whose mappings are explicitly unverified.

**Evaluation.** The ULB credit-card dataset (284,807 rows, 492 fraud, 0.173% prevalence; OpenML 1597) is
pushed through the real adapter → schema → classifier path. Missing Kavach fields are filled with neutral
constants that cannot trigger any social-engineering rule. Per-level precision / recall / F1 / confusion
matrix and AUC-PR are reported, with the domain mismatch stated first.

**Federated learning.** Online logistic regression per bank; only weight deltas leave a bank. FedAvg and
coordinate-median aggregation; versioned model store that refuses spec mismatches; a threshold bridge that
maps probability → L0–L3 with per-bank calibrated cutoffs and fails to L2 (human review), never L0. The
cold-start experiment partitions ULB into 5 synthetic banks under three splits — `iid` (control), `amount`
(banks skewed to amount-quantile bands, p=0.7) and `cluster` (k-means on V1..V5, p=0.7) — holds each bank
out in turn over 3 seeds (15 cells per split), sweeps the held-out bank's labelled history over
{0, 50, 500, 5000} rows, and compares local-only, FedAvg, coordinate-median and a pooled-data upper bound
on a fixed test set. AUC-PR is reported with lift over each cell's own prevalence.

**Hardening.** A sanitizer for the three free-text fields (NFKC, control/zero-width/bidi stripping, script-
agnostic allow-list, per-field clamp, seven instruction-pattern families, base64 detection), a renderer that
places surviving text only inside a delimited data block in the user turn, and a 44-case adversarial corpus.

## 3. Results

### 3.1 Pipeline mechanics (Component D)

| | AUC-PR | AUC-ROC |
|---|---|---|
| v1 rules (reconstructed thresholds; amount-only on this data) | 0.0018 | 0.535 |
| random ranker (= prevalence) | 0.0016 | 0.5 |
| reference logistic regression on V1..V28 + log(amount), held-out split | 0.760 | — |

282,982 of 284,807 rows passed canonical validation; 1,825 zero-amount rows were rejected and counted, 27 of
them fraud. Only three rule branches ever fired (`L0_default`, `L1_low_amount`, `L2_high_amount_alone`),
confirming that with neutral fills the v1 rules reduce to amount thresholds. Acting on ≥L1 flags 18.47% of
traffic for 25.38% recall; acting on ≥L2 flags 0.87% for 1.72% recall. The rules are indistinguishable from
random on card fraud; the reference model shows the dataset is separable. The gap is missing signal, not
broken mechanics — this dataset carries none of Kavach's inputs.

### 3.2 Federated cold start (Component B, Result 1)

Headline — n=5000, the only history size at which every held-out bank's local model had fraud rows to learn
from (minimum 2 under `amount`, 1 under `cluster`, 6 under `iid`):

| split | local-only | FedAvg | coord-median | pooled | FedAvg beats local | gap closed |
|---|---|---|---|---|---|---|
| `amount` | 0.581 | **0.679** | 0.669 | 0.721 | 10/15 | 70% |
| `cluster` | 0.579 | 0.659 | **0.668** | 0.705 | 14/15 | 64% |
| `iid` | 0.631 | **0.692** | 0.691 | 0.728 | 14/15 | 63% |

At n=50, 0–2 of 15 cells contained any fraud row; at n=500, 7–9. Conditioning n=500 on that:

| split | condition | cells | local-only | FedAvg | FedAvg beats local |
|---|---|---|---|---|---|
| `amount` | ≥1 fraud row | 8 | 0.598 | 0.693 | 6/8 |
| `amount` | no fraud row | 7 | 0.014 | 0.656 | 7/7 |
| `cluster` | ≥1 fraud row | 7 | 0.664 | 0.703 | 4/7 |
| `cluster` | no fraud row | 8 | 0.011 | 0.608 | 8/8 |
| `iid` | ≥1 fraud row | 9 | 0.466 | 0.696 | 9/9 |
| `iid` | no fraud row | 6 | 0.008 | 0.720 | 6/6 |

One fraud label moves local-only from ≈0.01 to 0.47–0.66 AUC-PR. Federation's advantage is therefore driven
by positive-class scarcity, not sample scarcity: it supplies the ranking direction that only positive examples
teach, from banks that have them. Where the local model has seen a handful of frauds the advantage is real but
modest (+0.06 to +0.10 at n=5000). The win holds on lift as well as raw AUC-PR; paired win counts are
identical under both by construction. Under `cluster`, coordinate-median beats FedAvg with no attacker in
10–13 of 15 cells at every history size; under `amount` it costs at most 0.011.

### 3.3 Poisoning (Result 2)

`amount` split, 4 contributors, attacker submits −10× inverted or N(0,5²) garbage deltas with a 20× inflated
sample count; held-out bank then fine-tunes on 0 / 500 / 5000 own rows.

| attackers | attack | FedAvg n=0 | median n=0 | FedAvg n=5000 | median n=5000 |
|---|---|---|---|---|---|
| 0 | — | 0.682 | 0.682 | 0.688 | 0.680 |
| 1 | inverted | 0.001 | 0.691 | 0.001 | 0.680 |
| 1 | garbage | 0.003 | 0.689 | 0.010 | 0.681 |
| 2 of 4 | inverted | 0.001 | 0.001 | 0.001 | 0.001 |
| 2 of 4 | garbage | 0.012 | 0.349 | 0.039 | 0.441 |

One attacker destroys FedAvg and does not move the median. Local fine-tuning on 5000 rows (mean 9.1 fraud
labels) recovers 0% of FedAvg's loss under the inverted attack and 4% under two garbage attackers — the
realistic operating case is not more forgiving than the attack-favourable one. Two of four attackers is the
median's breakdown point and every rule fails there.

### 3.4 Threshold bridge (Result 3)

One FedAvg model at every bank; only cutoffs differ (plus a full-stack row with per-bank fine-tuning). ≥L2
alert budget 1% of each bank's traffic; 15 bank-cells per split.

| split | per-bank L2 cutoffs differ by | global cutoff alert volume | per-bank | full stack | pooled ≥L2 recall: global / per-bank / full stack |
|---|---|---|---|---|---|
| `amount` | 2.5× | 0.37–3.11% | 0.86–1.14% | 0.79–1.23% | 86.1 / 85.7 / 85.3% |
| `cluster` | 3.8× | 0.33–2.64% | 0.85–1.17% | 0.85–1.19% | 85.4 / 83.5 / 85.4% |
| `iid` | 1.1× | 0.89–1.19% | 0.82–1.26% | 0.84–1.26% | 85.6 / 85.6 / 84.2% |

Per-bank calibration holds each bank's alert volume at budget where a global cutoff sends one bank 3.11% and
another 0.37%. It does not catch more fraud: for the same total alerts, pooled recall is within 0.4 pts under
`amount` and 1.9 pts lower under `cluster`. Labelled precision-target calibration meets its ≥L2 target in
25/45 bank-cells per-bank vs 22/45 with the global cutoff, and which is better flips by split.

### 3.5 Prompt injection (§5.2)

44/44 adversarial cases handled: 27 targeting `remarks`, 3 `device_location`, 3 `receiver_name`, 11 benign
Hindi/Hinglish/English controls passing unflagged (0 false positives). System prompt byte-identical in every
case; exactly one data-block delimiter pair in every rendered prompt.

## 4. Limitations

1. **Domain.** All evaluation is on 2013 European card-present fraud with anonymised PCA features. None of
   Kavach's social-engineering signals exist in it. Every number above is a property of the pipeline and the
   federated machinery, not of fraud detection on Indian UPI/IMPS traffic.
2. **Synthetic banks.** Partitions of one dataset with a mixing matrix. Real banks differ in fraud typology,
   labelling latency and base rate in ways this does not capture.
3. **Degenerate baselines.** n=0 and most n=50 cells compare against a local model with no positive
   examples; those comparisons are reported as the degenerate reference, not as results.
4. **Simple learner.** Logistic regression everywhere; a stronger local learner would shrink the n=5000 gap.
5. **Crude attackers.** Sign-flip, noise and count inflation. A small consistent bias each round is not tested
   and a median does not defeat it. No differential privacy or secure aggregation.
6. **Reconstructed rules.** v1 thresholds were rebuilt from the brief; the n8n export was unavailable.
7. **Unverified integrations.** Adapter mappings, the RBI FR-2 field structure and the voice-liveness
   interface are templates and stubs, labelled as such in code; none has been checked against a real API,
   the current RBI Master Direction, or any detector.
8. **No LLM exercised.** The sanitizer is measured on what reaches the prompt, not on what the model does
   with plain-language persuasion that survives it.

## 5. Future work

- A labelled UPI/IMPS social-engineering dataset with the fields Kavach actually uses — the single change
  that would turn §3.1 from a mechanics check into evidence.
- Drop the exported v1 thresholds into `RuleConfig` and regenerate the eval report.
- Live-model red-teaming of the prompt renderer for plain-language persuasion.
- Stealthy poisoning (small consistent bias), and pre-adoption scoring of an incoming global model on local
  history as the §5.5 detection control the Result 2 finding calls for.
- Secure aggregation / differential privacy on deltas, so "no raw data leaves the bank" becomes "nothing
  about the bank's data can be inferred from its update".
- Compliance review of the FR-2 template and one real adapter written against a bank's actual API document.
