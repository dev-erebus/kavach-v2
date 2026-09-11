# Kavach v2 — Federated Cold-Start Experiment

## What this is and is not

**Is:** a test of the *mechanism* in Component B — whether aggregating weight deltas from several
partitions transfers useful structure to a partition that contributed nothing, and how fast local
history closes the gap. Run on the ULB credit-card dataset (284,807 rows, fraud prevalence
0.173%), partitioned into 5 synthetic "banks".

**Is not:** evidence about Indian UPI/IMPS fraud. The data is 2013 European card fraud with
anonymised PCA features; the "banks" are partitions, not institutions; the feature spec is
`ulb_creditcard_eval@v1[29]` (evaluation only), not the production `FEATURE_CONTRACT`. Read every number below
as a property of the federated-learning machinery, not of Kavach's fraud detection.

## Protocol

- **Learner:** online logistic regression (numpy SGD), fixed transforms only, `pos_weight=50.0`,
  `lr=0.05`, `l2=0.0001`, batch 256. Identical for every method.
- **Federation:** 5 rounds; each contributing bank trains 1 local epoch(s) from the
  current global model and sends only `weights_after − weights_before` with its row count. Aggregated by
  FedAvg (sample-weighted mean) or coordinate-median. Delta norms clipped at 50 in both cases.
- **Held-out bank:** every bank in turn. Its rows are permuted once per seed; the first *n* form its
  labelled history (n ∈ [0, 50, 500, 5000]), rows beyond position 5000 form a **fixed test set** so
  all history sizes are scored on identical rows.
- **Methods compared on the held-out bank**, each followed by the *same* 10-epoch fine-tune on the *n* history rows:
  - `local_only` — from zero weights. At n=0 this is a constant scorer (AUC-PR = prevalence).
  - `fedavg` — FedAvg global model learned from the other 4 banks' deltas.
  - `coord_median` — same, coordinate-median aggregation.
  - `pooled` — the other banks' raw rows pooled centrally, 5 epochs. **Upper bound**: what federation
    is trying to approach without anyone pooling data.
- **Seeds:** [0, 1, 2] → 15 (seed × held-out bank) cells per split per history size.
- **Metric:** AUC-PR on the held-out bank's test rows. Spread is the sd over cells.

## Splits

### `iid`

IID control: each row assigned to a bank uniformly at random.

| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |
|---|---|---|---|---|
| bank0 | 56,933 | 105 | 0.184% | 22.10 |
| bank1 | 56,757 | 107 | 0.189% | 21.99 |
| bank2 | 57,050 | 104 | 0.182% | 21.84 |
| bank3 | 57,133 | 97 | 0.170% | 21.95 |
| bank4 | 56,934 | 79 | 0.139% | 22.36 |

### `amount`

Amount skew: rows binned into 5 amount-quantile bands; a row in band k goes to bank k with p=0.7 and to each other bank with p=0.075. Banks therefore differ in transaction-size distribution AND fraud prevalence.

| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |
|---|---|---|---|---|
| bank0 | 56,836 | 178 | 0.313% | 1.98 |
| bank1 | 57,292 | 60 | 0.105% | 9.83 |
| bank2 | 57,129 | 55 | 0.096% | 22.00 |
| bank3 | 56,591 | 88 | 0.156% | 52.90 |
| bank4 | 56,959 | 111 | 0.195% | 146.48 |

### `cluster`

Feature-cluster skew: k-means (k=5) on V1..V5, the five highest-variance PCA components; a row in cluster k goes to bank k with p=0.7, else uniformly elsewhere. Banks occupy different regions of feature space.

| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |
|---|---|---|---|---|
| bank0 | 85,722 | 66 | 0.077% | 29.42 |
| bank1 | 62,653 | 140 | 0.223% | 20.24 |
| bank2 | 23,477 | 43 | 0.183% | 27.90 |
| bank3 | 88,090 | 56 | 0.064% | 17.68 |
| bank4 | 24,865 | 187 | 0.752% | 19.95 |

## Result 1 — cold-start curves

![cold start](cold_start_curves.png)

### `iid` — AUC-PR on held-out bank, mean ± sd (n cells)

| History rows | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | fedavg vs pooled (gap) |
|---|---|---|---|---|---|---|
| 0 | 0.002 ± 0.000 | 0.718 ± 0.043 | 0.717 ± 0.043 | 0.740 ± 0.045 | 15/15 | +0.022 |
| 50 | 0.081 ± 0.157 | 0.696 ± 0.093 | 0.697 ± 0.096 | 0.725 ± 0.085 | 15/15 | +0.029 |
| 500 | 0.283 ± 0.308 | 0.705 ± 0.065 | 0.703 ± 0.067 | 0.725 ± 0.070 | 15/15 | +0.020 |
| 5000 | 0.631 ± 0.092 | 0.692 ± 0.062 | 0.691 ± 0.062 | 0.728 ± 0.057 | 14/15 | +0.036 |

### `amount` — AUC-PR on held-out bank, mean ± sd (n cells)

| History rows | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | fedavg vs pooled (gap) |
|---|---|---|---|---|---|---|
| 0 | 0.002 ± 0.001 | 0.671 ± 0.103 | 0.664 ± 0.098 | 0.699 ± 0.100 | 15/15 | +0.028 |
| 50 | 0.016 ± 0.018 | 0.672 ± 0.101 | 0.665 ± 0.097 | 0.705 ± 0.096 | 15/15 | +0.033 |
| 500 | 0.326 ± 0.348 | 0.676 ± 0.090 | 0.668 ± 0.083 | 0.706 ± 0.089 | 13/15 | +0.030 |
| 5000 | 0.581 ± 0.211 | 0.679 ± 0.076 | 0.669 ± 0.076 | 0.721 ± 0.066 | 10/15 | +0.041 |

### `cluster` — AUC-PR on held-out bank, mean ± sd (n cells)

| History rows | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | fedavg vs pooled (gap) |
|---|---|---|---|---|---|---|
| 0 | 0.003 ± 0.003 | 0.593 ± 0.186 | 0.616 ± 0.185 | 0.658 ± 0.157 | 15/15 | +0.065 |
| 50 | 0.107 ± 0.255 | 0.620 ± 0.154 | 0.638 ± 0.155 | 0.682 ± 0.125 | 14/15 | +0.062 |
| 500 | 0.316 ± 0.361 | 0.652 ± 0.125 | 0.666 ± 0.121 | 0.710 ± 0.105 | 12/15 | +0.057 |
| 5000 | 0.579 ± 0.176 | 0.659 ± 0.121 | 0.668 ± 0.114 | 0.705 ± 0.100 | 14/15 | +0.046 |

### Reading the curves

- **`iid`:** at 0 history, federated 0.718 vs local-only 0.002 (pooled upper bound 0.740); federated wins 15/15 cells. At 5000 history, federated 0.692 vs local-only 0.631, wins 14/15. Federation captures 97% of the pooled-vs-local gap at cold start.
- **`amount`:** at 0 history, federated 0.671 vs local-only 0.002 (pooled upper bound 0.699); federated wins 15/15 cells. At 5000 history, federated 0.679 vs local-only 0.581, wins 10/15. Federation captures 96% of the pooled-vs-local gap at cold start.
- **`cluster`:** at 0 history, federated 0.593 vs local-only 0.003 (pooled upper bound 0.658); federated wins 15/15 cells. At 5000 history, federated 0.659 vs local-only 0.579, wins 14/15. Federation captures 90% of the pooled-vs-local gap at cold start.

**FedAvg vs coordinate-median with no attacker** (the efficiency price of robustness, if any):

- `iid`: n=0: -0.001 (median wins 7/15); n=50: +0.000 (median wins 7/15); n=500: -0.002 (median wins 4/15); n=5000: -0.001 (median wins 6/15)
- `amount`: n=0: -0.006 (median wins 2/15); n=50: -0.007 (median wins 2/15); n=500: -0.008 (median wins 3/15); n=5000: -0.011 (median wins 1/15)
- `cluster`: n=0: +0.023 (median wins 13/15); n=50: +0.018 (median wins 12/15); n=500: +0.014 (median wins 12/15); n=5000: +0.009 (median wins 10/15)

A negative number is the cost of using the median when everyone is honest; a positive one means the median generalised *better* to the held-out bank — which happens when partitions are skewed enough that the sample-weighted mean is dominated by whichever contributor is largest or most atypical.

## Result 2 — poisoned deltas (§5.4)

Split `amount`, every bank held out in turn, seeds [0, 1, 2], **0 local history** (pure
cold start, so the global model is all the held-out bank has). Of the 4 contributing banks,
0, 1 or 2 are attackers. An attacker replaces its honest delta with either **inverted** (−10× the
honest delta) or **garbage** (N(0, 5²) noise), and **claims 20× its true sample count** to dominate a
weighted average. Delta norm clipping (50) applies to all rules.

![poisoning](poisoning.png)

| Attackers | Attack | `fedavg` | `trimmed_mean_20` | `coord_median` |
|---|---|---|---|---|
| 0 | none | 0.680 ± 0.099 | 0.681 ± 0.088 | 0.681 ± 0.088 |
| 1 | inverted | 0.001 ± 0.001 | 0.689 ± 0.093 | 0.689 ± 0.093 |
| 1 | garbage | 0.003 ± 0.002 | 0.686 ± 0.096 | 0.686 ± 0.096 |
| 2 | inverted | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.001 ± 0.001 |
| 2 | garbage | 0.012 ± 0.009 | 0.346 ± 0.119 | 0.346 ± 0.119 |

Coordinate-median tolerates strictly fewer than half of the participants being malicious. With
4 contributors that means **1 attacker is inside its breakdown point and 2 is at it** — the
2-attacker rows are expected to show degradation for *every* rule, and they are reported precisely
because they are the honest edge of the guarantee. With 20% trimming on 4 participants the trimmed
mean removes 0 values per side and falls back to the median, so `trimmed_mean_20` ≈ `coord_median`
here; it separates from it only with more participants.

## Limitations

1. Card fraud, not UPI social engineering; PCA features, not Kavach's contract. Mechanism result only.
2. "Banks" are synthetic partitions of one dataset. Real banks differ in ways a mixing matrix does not capture
   (different fraud typologies, different labelling latency and quality, different base rates by an order of magnitude).
3. Logistic regression is a deliberately simple learner. The federated-vs-local *gap* is what is being measured, and a
   stronger local learner would shrink it at large history — the cold-start end of the curve is the claim, not the right end.
4. Attackers here are crude (sign flip, noise, inflated counts). A stealthy attacker who shifts deltas by a small,
   consistent amount every round is not tested and is *not* defeated by a median.
5. No differential privacy or secure aggregation: the aggregator sees each bank's delta in the clear. "No raw data leaves
   the bank" is true; "nothing about the bank's data can be inferred from its delta" is not claimed.
6. Fine-tune epochs, rounds and learning rate were set once, not tuned per method. Tuning would move numbers, not the shape.

_Run time 0.0 min. Raw trials in `cold_start_trials.json`._
