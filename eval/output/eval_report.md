# Kavach v2 — Evaluation Report

_Generated 2026-09-11 18:22 UTC_

## ⚠️ Read this first — what these numbers do and do not show

**This dataset is card-present / e-commerce card fraud. It is not UPI/IMPS social-engineering fraud.**

The `ulb` dataset (OpenML dataset id 1597 'creditcard' v1 (mirror of the ULB/Kaggle dataset; fetched in 5s; OpenML's copy omits the 'Time' column)) contains anonymised card transactions with a binary fraud
label. It has **no phone-call context, no sender age, no beneficiary-account age, no device location,
no voice-liveness signal** — none of the features Kavach's Level 2/3 logic exists to exploit. Real
Indian UPI/IMPS scam fraud (digital-arrest calls, fake-KYC, elderly-targeted mule transfers) has a
completely different signal profile and is not represented here at all.

**What this evaluation therefore validates:** that the pipeline mechanics work end to end —
adapter → canonical schema validation → rule classifier → per-level metrics — on 282,982 labelled rows
without silent drops, and what a v1-style amount-driven rule set does on real (if foreign-domain)
fraud labels.

**What it does not validate:** that Kavach detects the fraud it is designed for. Every number below
should be read as a *pipeline mechanics* result, not a *fraud detection* result. Do not quote these
figures as Kavach's precision or recall on Indian banking fraud.

**How the missing fields were handled.** To send rows through the real adapter/schema path, the
absent canonical fields were filled with fixed *neutral* values that cannot trigger any
social-engineering rule: sender age 40, receiver account age
3650 days, no active phone call, biometric verdict *unknown*.
As a consequence **the v1 rules degrade to an amount-only classifier on this data**; the rule
attribution table below confirms that only amount branches fire. Synthesising fraud-correlated
values for the missing fields would have manufactured evidence, so it was not done.

**Currency.** Amounts are in EUR and were placed on the INR rule scale with a single fixed constant
(1 EUR = ₹90). This only positions the amount thresholds; AUC-PR and AUC-ROC are
rank-based and unaffected by it.

**Rule provenance.** The v1 thresholds below are a *reconstruction* from the build brief, because the
n8n Switch-node export was not available. They are held in `core/rules.py::RuleConfig` so the real
values can be dropped in and this report regenerated.

## Dataset

- **Name:** `ulb`
- **Provenance:** OpenML dataset id 1597 'creditcard' v1 (mirror of the ULB/Kaggle dataset; fetched in 5s; OpenML's copy omits the 'Time' column)
- **Rows:** 284,807
- **Fraud rows:** 492 (0.17%)
- Amounts are in EUR (European cardholders, Sept 2013). Features V1..V28 are anonymised PCA components.

### Canonical-schema validation (pipeline acceptance criterion: 100% of fields validated)

- Rows accepted by `KavachTransaction`: **282,982** of 284,807
- Rows rejected (counted, not silently dropped): **1,825**, of which **27 were labelled fraud** — those frauds are excluded from every metric below, which flatters recall by at most 27/492 of the fraud class.
  - 1,825 × `schema_violation: amount_inr: Value error, amount_inr must be > 0`

## v1 rule configuration used (reconstructed)

| Parameter | Value |
|---|---|
| amount_high_inr (L2 alone / L3 with signals) | ₹100,000 |
| amount_medium_inr | ₹50,000 |
| amount_low_inr (L1) | ₹10,000 |
| elderly_age | 60 |
| fresh_receiver_days | 7 |
| young_receiver_days | 30 |

## Results — v1 rules on all 282,982 validated rows

- **AUC-PR:** 0.0018 (random ranker: 0.0016; lift ×1.1)
- **AUC-ROC:** 0.5351
- _Score used for ranking is the ordinal level itself (4 distinct values), so the PR curve has at most four corners. This coarseness is a property of a rule pipeline, not of the metric._

### Per-operating-point metrics

Each row is the binary classifier 'act if level ≥ X' — the three ways the pipeline can actually be wired.

| Operating point | Precision | Recall | F1 | Flag rate | FPR | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| act on ≥L1 | 0.23% | 25.38% | 0.004 | 18.47% | 18.45% | 118 | 52,138 | 347 | 230,379 |
| act on ≥L2 | 0.33% | 1.72% | 0.005 | 0.87% | 0.87% | 8 | 2,446 | 457 | 280,071 |
| act on ≥L3 | n/a | 0.00% | 0.000 | 0.00% | 0.00% | 0 | 0 | 465 | 282,517 |

![confusion matrices](confusion_matrices.png)

### Level distribution by class

| Level | Legit | Fraud | Fraud share at this level |
|---|---|---|---|
| L0 (allow) | 230,379 | 347 | 0.15% |
| L1 (warn) | 49,692 | 110 | 0.22% |
| L2 (hold_and_verify) | 2,446 | 8 | 0.33% |
| L3 (hard_lock_and_review) | 0 | 0 | n/a |

![level distribution](level_distribution.png)

### Which rule fired (false-positive breakdown)

| Rule that fired | Rows | Fraud | Legit | Precision of rule |
|---|---|---|---|---|
| `L0_default` | 230,726 | 347 | 230,379 | 0.15% |
| `L1_low_amount` | 49,802 | 110 | 49,692 | 0.22% |
| `L2_high_amount_alone` | 2,454 | 8 | 2,446 | 0.33% |

## Reference point — a plain ML model on the dataset's own features

_This is **not** a Kavach component._ It answers one question: is the rule pipeline's score
low because the data is hard, or because the rules can't see the signal? `logistic regression on V1..V28 + log(amount)` — StandardScaler → class-balanced logistic regression, default regularisation, no tuning —
is trained on 198,087 rows and evaluated on a stratified held-out split of 84,895 rows.
The v1 rules are re-scored on that same split so the comparison is like for like.

| Model | AUC-PR | AUC-ROC | Best-F1 operating point |
|---|---|---|---|
| v1 rules (act on ≥L2) | 0.0017 | 0.5152 | 0.004 |
| logistic regression on V1..V28 + log(amount) | 0.7596 | 0.9881 | 0.688 |

The gap between the two rows is the information in features the rule pipeline cannot see. On Kavach's
real domain the analogous features are the social-engineering signals — which is exactly what
Component B (federated per-bank learning) is meant to exploit, and why this baseline is the
comparison point for it.

![pr curves](pr_curve.png)

## Limitations (recap)

1. Wrong fraud domain: card-present/e-commerce, European cardholders, 2013. Not UPI/IMPS, not India, not social engineering.
2. Missing canonical fields filled with neutral constants; the rules therefore reduce to amount thresholds here.
3. Rule thresholds are reconstructed, not the deployed v1 export.
4. EUR→INR is a single fixed constant chosen to exercise the thresholds, not a claim about purchasing-power equivalence.
5. The dataset's PCA features are anonymised; nothing about *why* the reference model separates fraud transfers to Kavach.
6. No LLM node was exercised. Node 3's classification is out of scope for this harness (it needs a live vLLM endpoint and, more importantly, a domain-appropriate dataset).
