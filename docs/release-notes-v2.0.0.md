Kavach v2.0.0 — first tagged release of the research prototype. This tag is the fixed artifact that
[`CITATION.cff`](CITATION.cff) refers to; results below are from the committed reports at this commit.

**Status:** research prototype. Every number is measured on the ULB card-fraud dataset (2013 European
card-present fraud) and validates *pipeline mechanics*, not fraud detection on Indian UPI/IMPS data.
Adapters, the FR-2 template and the voice-liveness interface are illustrative and unverified. Not for
production decisions.

## Four findings

1. **The v1 rules are indistinguishable from random on card fraud, and that is the correct result.**
   AUC-PR 0.0018 vs 0.0016 for a random ranker; a plain logistic regression on the dataset's own
   features reaches 0.760. The dataset carries none of Kavach's signals, so only amount rules can fire —
   the gap is missing signal, not broken mechanics.
2. **Federation's cold-start advantage comes from positive-class scarcity, not sample scarcity.**
   A local model with 500 rows and zero fraud labels scores ≈0.01 AUC-PR; one fraud label lifts it to
   0.47–0.66. At n=5000 (every held-out bank has fraud labels) federation still wins: 0.581 → 0.679 under
   the `amount` split (10/15 cells), 14/15 under `cluster` and `iid`, closing 63–70% of the pooled-vs-local
   gap. Comparisons at n=0 and most of n=50 are against a baseline that cannot rank and are reported as
   the degenerate reference, not the result.
3. **One poisoned participant collapses FedAvg from 0.682 to 0.001 AUC-PR; coordinate-median holds at
   0.691.** Local fine-tuning on 5,000 rows recovers 0% of FedAvg's loss. Two of four attackers is the
   median's breakdown point and it fails there.
4. **Per-bank calibrated thresholds buy alert-budget adherence, not detection.** A single global cutoff
   sends 0.37–3.11% of a bank's traffic to review against a 1% budget; per-bank cutoffs hold every bank
   at 0.86–1.14% — for the same pooled recall (within 0.4 pts; 1.9 pts *worse* per-bank under `cluster`).

Prompt-injection hardening: 44/44 adversarial cases handled (27 on the UPI payer-typed `remarks` field),
11 benign Hindi/Hinglish/English controls unflagged, system prompt byte-identical in every case. Not
measured: plain-language persuasion against a live LLM.

Full write-up: [RESULTS.md](https://github.com/dev-erebus/kavach-v2/blob/v2.0.0/RESULTS.md).
Detailed reports: [`federated/output/cold_start_report.md`](https://github.com/dev-erebus/kavach-v2/blob/v2.0.0/federated/output/cold_start_report.md),
[`eval/output/eval_report.md`](https://github.com/dev-erebus/kavach-v2/blob/v2.0.0/eval/output/eval_report.md),
[`eval/output/adversarial_report.md`](https://github.com/dev-erebus/kavach-v2/blob/v2.0.0/eval/output/adversarial_report.md).

## What is in the release

- Component A — canonical schema, 17-slot `FEATURE_CONTRACT`, adapter registry, per-bank secrets,
  three illustrative adapters (Finacle / Temenos T24 / TCS BaNCS).
- Component D — evaluation harness on ULB via OpenML; §5.2 adversarial suite.
- Component B — contract-agnostic online LR, FedAvg + coordinate-median + trimmed-mean, versioned model
  store, threshold bridge (fails to L2, never L0), `/feedback` contract; cold-start, poisoning and
  threshold-bridge experiments.
- §5.2 sanitizer and data-block prompt renderer; §5.1 voice-liveness interface + stub.
- Component C — RBI FR-2 template generator (`filing_ready: false` until compliance review).
- 145 offline tests; fully pinned `requirements.txt`.

## Reproduce

```bash
git clone --branch v2.0.0 https://github.com/dev-erebus/kavach-v2.git && cd kavach-v2
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pytest -q
python -m eval.run_eval
python -m federated.experiments.cold_start
```

## Authorship

Claude code was used to write the code under the author's supervision and instructions
