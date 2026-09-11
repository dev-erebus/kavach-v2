# Kavach v2 — Federated Cold-Start Experiment

## What this is and is not

**Is:** a test of the *mechanism* in Component B — whether aggregating weight deltas from several
partitions transfers useful structure to a partition that contributed nothing, how fast local
history closes the gap, and (Result 3) whether per-bank calibrated cutoffs do anything a single
global cutoff does not. Run on the ULB credit-card dataset (284,807 rows, fraud prevalence
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
  - `local_only` — from zero weights. **At n=0 this is a constant scorer** — every row gets the same probability,
    so it cannot rank and its AUC-PR equals the prevalence (lift exactly 1.0). Comparisons at n=0 are against a
    model that cannot do the task; they are reported as the degenerate reference, not as the result.
  - `fedavg` — FedAvg global model learned from the other 4 banks' deltas.
  - `coord_median` — same, coordinate-median aggregation.
  - `pooled` — the other banks' raw rows pooled centrally, 5 epochs. **Upper bound**: what federation
    is trying to approach without anyone pooling data.
- **Seeds:** [0, 1, 2] → 15 (seed × held-out bank) cells per split per history size.
- **Metrics:** AUC-PR on the held-out bank's test rows, **and lift = AUC-PR / that cell's own prevalence.**
  AUC-PR's floor is the prevalence, and held-out prevalence varies several-fold across banks under the non-IID
  splits, so a raw cross-bank mean mixes scales and its sd includes base-rate variation. Lift puts every cell on
  its own floor. Paired win counts are identical under both metrics (same test set per cell); lift changes the
  cross-bank mean and spread. Spread is the sd over cells.

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

Prevalence ratio across banks (max/min): **1.4×**.

### `amount`

Amount skew: rows binned into 5 amount-quantile bands; a row in band k goes to bank k with p=0.7 and to each other bank with p=0.075. Banks therefore differ in transaction-size distribution AND fraud prevalence.

| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |
|---|---|---|---|---|
| bank0 | 56,836 | 178 | 0.313% | 1.98 |
| bank1 | 57,292 | 60 | 0.105% | 9.83 |
| bank2 | 57,129 | 55 | 0.096% | 22.00 |
| bank3 | 56,591 | 88 | 0.156% | 52.90 |
| bank4 | 56,959 | 111 | 0.195% | 146.48 |

Prevalence ratio across banks (max/min): **3.3×**.

### `cluster`

Feature-cluster skew: k-means (k=5) on V1..V5, the five highest-variance PCA components; a row in cluster k goes to bank k with p=0.7, else uniformly elsewhere. Banks occupy different regions of feature space.

| Bank | Rows | Fraud | Prevalence | Median amount (EUR) |
|---|---|---|---|---|
| bank0 | 85,722 | 66 | 0.077% | 29.42 |
| bank1 | 62,653 | 140 | 0.223% | 20.24 |
| bank2 | 23,477 | 43 | 0.183% | 27.90 |
| bank3 | 88,090 | 56 | 0.064% | 17.68 |
| bank4 | 24,865 | 187 | 0.752% | 19.95 |

Prevalence ratio across banks (max/min): **11.8×**.

## Result 1 — cold-start curves

Raw AUC-PR (left figure) and lift over each cell's prevalence floor (right figure, log scale).

![cold start](cold_start_curves.png)

![cold start lift](cold_start_lift.png)

### `iid` — held-out bank, mean ± sd over 15 cells; each cell shows AUC-PR and lift

| History rows | fraud rows in history (mean) | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | pooled − fedavg (AUC-PR) |
|---|---|---|---|---|---|---|---|
| 50 | 0.1 | 0.081 ± 0.157<br><sub>lift 48 ± 94</sub> | 0.696 ± 0.093<br><sub>lift 410 ± 64</sub> | 0.697 ± 0.096<br><sub>lift 410 ± 65</sub> | 0.725 ± 0.085<br><sub>lift 426 ± 61</sub> | 15/15 | +0.029 |
| 500 | 0.9 | 0.283 ± 0.308<br><sub>lift 168 ± 186</sub> | 0.705 ± 0.065<br><sub>lift 414 ± 46</sub> | 0.703 ± 0.067<br><sub>lift 413 ± 46</sub> | 0.725 ± 0.070<br><sub>lift 426 ± 48</sub> | 15/15 | +0.020 |
| 5000 | 9.4 | 0.631 ± 0.092<br><sub>lift 369 ± 43</sub> | 0.692 ± 0.062<br><sub>lift 406 ± 43</sub> | 0.691 ± 0.062<br><sub>lift 405 ± 42</sub> | 0.728 ± 0.057<br><sub>lift 428 ± 44</sub> | 14/15 | +0.036 |
| 0 — *degenerate reference* | 0.0 | 0.002 ± 0.000<br><sub>lift 1 ± 0</sub> | 0.718 ± 0.043<br><sub>lift 422 ± 40</sub> | 0.717 ± 0.043<br><sub>lift 422 ± 40</sub> | 0.740 ± 0.045<br><sub>lift 435 ± 43</sub> | 15/15 | +0.022 |

### `amount` — held-out bank, mean ± sd over 15 cells; each cell shows AUC-PR and lift

| History rows | fraud rows in history (mean) | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | pooled − fedavg (AUC-PR) |
|---|---|---|---|---|---|---|---|
| 50 | 0.0 | 0.016 ± 0.018<br><sub>lift 11 ± 14</sub> | 0.672 ± 0.101<br><sub>lift 476 ± 209</sub> | 0.665 ± 0.097<br><sub>lift 469 ± 199</sub> | 0.705 ± 0.096<br><sub>lift 502 ± 231</sub> | 15/15 | +0.033 |
| 500 | 1.1 | 0.326 ± 0.348<br><sub>lift 183 ± 222</sub> | 0.676 ± 0.090<br><sub>lift 478 ± 208</sub> | 0.668 ± 0.083<br><sub>lift 470 ± 195</sub> | 0.706 ± 0.089<br><sub>lift 503 ± 230</sub> | 13/15 | +0.030 |
| 5000 | 10.1 | 0.581 ± 0.211<br><sub>lift 372 ± 171</sub> | 0.679 ± 0.076<br><sub>lift 475 ± 193</sub> | 0.669 ± 0.076<br><sub>lift 466 ± 186</sub> | 0.721 ± 0.066<br><sub>lift 508 ± 215</sub> | 10/15 | +0.041 |
| 0 — *degenerate reference* | 0.0 | 0.002 ± 0.001<br><sub>lift 1 ± 0</sub> | 0.671 ± 0.103<br><sub>lift 475 ± 210</sub> | 0.664 ± 0.098<br><sub>lift 467 ± 196</sub> | 0.699 ± 0.100<br><sub>lift 499 ± 233</sub> | 15/15 | +0.028 |

### `cluster` — held-out bank, mean ± sd over 15 cells; each cell shows AUC-PR and lift

| History rows | fraud rows in history (mean) | `local_only` | `fedavg` | `coord_median` | `pooled` | fedavg beats local (cells) | pooled − fedavg (AUC-PR) |
|---|---|---|---|---|---|---|---|
| 50 | 0.1 | 0.107 ± 0.255<br><sub>lift 38 ± 81</sub> | 0.620 ± 0.154<br><sub>lift 498 ± 388</sub> | 0.638 ± 0.155<br><sub>lift 518 ± 406</sub> | 0.682 ± 0.125<br><sub>lift 557 ± 430</sub> | 14/15 | +0.062 |
| 500 | 1.6 | 0.316 ± 0.361<br><sub>lift 102 ± 121</sub> | 0.652 ± 0.125<br><sub>lift 518 ± 382</sub> | 0.666 ± 0.121<br><sub>lift 533 ± 396</sub> | 0.710 ± 0.105<br><sub>lift 573 ± 421</sub> | 12/15 | +0.057 |
| 5000 | 13.1 | 0.579 ± 0.176<br><sub>lift 417 ± 284</sub> | 0.659 ± 0.121<br><sub>lift 514 ± 365</sub> | 0.668 ± 0.114<br><sub>lift 524 ± 374</sub> | 0.705 ± 0.100<br><sub>lift 568 ± 417</sub> | 14/15 | +0.046 |
| 0 — *degenerate reference* | 0.0 | 0.003 ± 0.003<br><sub>lift 1 ± 0</sub> | 0.593 ± 0.186<br><sub>lift 484 ± 399</sub> | 0.616 ± 0.185<br><sub>lift 507 ± 417</sub> | 0.658 ± 0.157<br><sub>lift 546 ± 439</sub> | 15/15 | +0.065 |

### Reading the curves

The result is the **n=50 and n=500 rows**: there the local-only model has real labelled data — on average 0.1 fraud rows at n=50, 0.9 fraud rows at n=500 — and is genuinely trying to rank.

- **`iid`:** n=50: federated 0.696 vs local 0.081 AUC-PR (lift 410× vs 48×), wins 15/15; n=500: federated 0.705 vs local 0.283 AUC-PR (lift 414× vs 168×), wins 15/15; n=5000: federated 0.692 vs local 0.631 AUC-PR (lift 406× vs 369×), wins 14/15. At n=50 federation captures 96% of the pooled-vs-local gap.
- **`amount`:** n=50: federated 0.672 vs local 0.016 AUC-PR (lift 476× vs 11×), wins 15/15; n=500: federated 0.676 vs local 0.326 AUC-PR (lift 478× vs 183×), wins 13/15; n=5000: federated 0.679 vs local 0.581 AUC-PR (lift 475× vs 372×), wins 10/15. At n=50 federation captures 95% of the pooled-vs-local gap.
- **`cluster`:** n=50: federated 0.620 vs local 0.107 AUC-PR (lift 498× vs 38×), wins 14/15; n=500: federated 0.652 vs local 0.316 AUC-PR (lift 518× vs 102×), wins 12/15; n=5000: federated 0.659 vs local 0.579 AUC-PR (lift 514× vs 417×), wins 14/15. At n=50 federation captures 89% of the pooled-vs-local gap.

**Does the win hold on lift?** Per-cell win counts are the same under lift by construction (shared test set). The cross-bank *means* are what lift changes, and the federated-over-local ordering of means is preserved in every split and every history size above — the claim does not depend on which scale is used.

**n=0 is the degenerate reference.** `local_only` at n=0 is a constant scorer (lift 1.0 in every cell); the 15/15/15 of 15 wins there say only that a model beats no model. They are not the headline.

### Per-held-out-bank breakdown (attributing the spread)

Mean over seeds for each held-out bank. `prev` is that bank's test-set prevalence — the AUC-PR floor for its row. Where AUC-PR tracks `prev` down a column, the cross-bank sd is base rate, not method.

#### `iid`

| Held-out | prev | `local_only` n=50 | `fedavg` n=50 | `coord_median` n=50 | `pooled` n=50 | `local_only` n=500 | `fedavg` n=500 | `coord_median` n=500 | `pooled` n=500 |
|---|---|---|---|---|---|---|---|---|---|
| bank0 | 0.171% | 0.015<br><sub>9×</sub> | 0.683<br><sub>401×</sub> | 0.684<br><sub>402×</sub> | 0.715<br><sub>420×</sub> | 0.220<br><sub>119×</sub> | 0.686<br><sub>404×</sub> | 0.683<br><sub>401×</sub> | 0.714<br><sub>419×</sub> |
| bank1 | 0.180% | 0.081<br><sub>46×</sub> | 0.740<br><sub>411×</sub> | 0.743<br><sub>412×</sub> | 0.765<br><sub>425×</sub> | 0.011<br><sub>6×</sub> | 0.744<br><sub>413×</sub> | 0.744<br><sub>413×</sub> | 0.767<br><sub>426×</sub> |
| bank2 | 0.175% | 0.062<br><sub>34×</sub> | 0.753<br><sub>439×</sub> | 0.754<br><sub>438×</sub> | 0.792<br><sub>461×</sub> | 0.594<br><sub>348×</sub> | 0.749<br><sub>437×</sub> | 0.750<br><sub>437×</sub> | 0.787<br><sub>460×</sub> |
| bank3 | 0.171% | 0.019<br><sub>12×</sub> | 0.726<br><sub>428×</sub> | 0.726<br><sub>428×</sub> | 0.733<br><sub>431×</sub> | 0.391<br><sub>248×</sub> | 0.726<br><sub>428×</sub> | 0.726<br><sub>428×</sub> | 0.732<br><sub>430×</sub> |
| bank4 | 0.160% | 0.229<br><sub>139×</sub> | 0.579<br><sub>369×</sub> | 0.577<br><sub>368×</sub> | 0.619<br><sub>395×</sub> | 0.197<br><sub>118×</sub> | 0.621<br><sub>390×</sub> | 0.613<br><sub>384×</sub> | 0.626<br><sub>393×</sub> |

Pearson r between a cell's AUC-PR and its prevalence at n=500 (15 cells): fedavg **+0.45**, local-only -0.05, pooled +0.47. A weak r means the spread is mostly method/seed variance, not base rate.

#### `amount`

| Held-out | prev | `local_only` n=50 | `fedavg` n=50 | `coord_median` n=50 | `pooled` n=50 | `local_only` n=500 | `fedavg` n=500 | `coord_median` n=500 | `pooled` n=500 |
|---|---|---|---|---|---|---|---|---|---|
| bank0 | 0.303% | 0.022<br><sub>7×</sub> | 0.781<br><sub>258×</sub> | 0.776<br><sub>256×</sub> | 0.800<br><sub>265×</sub> | 0.714<br><sub>236×</sub> | 0.778<br><sub>257×</sub> | 0.769<br><sub>254×</sub> | 0.799<br><sub>264×</sub> |
| bank1 | 0.111% | 0.006<br><sub>5×</sub> | 0.735<br><sub>666×</sub> | 0.724<br><sub>656×</sub> | 0.762<br><sub>692×</sub> | 0.231<br><sub>198×</sub> | 0.740<br><sub>671×</sub> | 0.728<br><sub>660×</sub> | 0.762<br><sub>691×</sub> |
| bank2 | 0.091% | 0.017<br><sub>20×</sub> | 0.658<br><sub>726×</sub> | 0.633<br><sub>698×</sub> | 0.740<br><sub>815×</sub> | 0.213<br><sub>238×</sub> | 0.657<br><sub>725×</sub> | 0.628<br><sub>693×</sub> | 0.739<br><sub>815×</sub> |
| bank3 | 0.140% | 0.022<br><sub>16×</sub> | 0.656<br><sub>473×</sub> | 0.653<br><sub>470×</sub> | 0.631<br><sub>454×</sub> | 0.013<br><sub>9×</sub> | 0.655<br><sub>471×</sub> | 0.651<br><sub>469×</sub> | 0.627<br><sub>450×</sub> |
| bank4 | 0.206% | 0.012<br><sub>6×</sub> | 0.527<br><sub>256×</sub> | 0.540<br><sub>262×</sub> | 0.591<br><sub>287×</sub> | 0.456<br><sub>232×</sub> | 0.548<br><sub>266×</sub> | 0.564<br><sub>274×</sub> | 0.603<br><sub>293×</sub> |

Pearson r between a cell's AUC-PR and its prevalence at n=500 (15 cells): fedavg **+0.23**, local-only +0.60, pooled +0.18. A weak r means the spread is mostly method/seed variance, not base rate.

#### `cluster`

| Held-out | prev | `local_only` n=50 | `fedavg` n=50 | `coord_median` n=50 | `pooled` n=50 | `local_only` n=500 | `fedavg` n=500 | `coord_median` n=500 | `pooled` n=500 |
|---|---|---|---|---|---|---|---|---|---|
| bank0 | 0.163% | 0.015<br><sub>15×</sub> | 0.563<br><sub>445×</sub> | 0.599<br><sub>472×</sub> | 0.631<br><sub>496×</sub> | 0.124<br><sub>64×</sub> | 0.558<br><sub>442×</sub> | 0.587<br><sub>461×</sub> | 0.624<br><sub>492×</sub> |
| bank1 | 0.116% | 0.016<br><sub>23×</sub> | 0.602<br><sub>741×</sub> | 0.632<br><sub>782×</sub> | 0.671<br><sub>834×</sub> | 0.171<br><sub>90×</sub> | 0.615<br><sub>763×</sub> | 0.638<br><sub>793×</sub> | 0.673<br><sub>836×</sub> |
| bank2 | 0.334% | 0.011<br><sub>10×</sub> | 0.617<br><sub>459×</sub> | 0.613<br><sub>452×</sub> | 0.672<br><sub>497×</sub> | 0.521<br><sub>173×</sub> | 0.744<br><sub>532×</sub> | 0.739<br><sub>526×</sub> | 0.779<br><sub>559×</sub> |
| bank3 | 0.113% | 0.004<br><sub>5×</sub> | 0.547<br><sub>673×</sub> | 0.575<br><sub>713×</sub> | 0.648<br><sub>784×</sub> | 0.009<br><sub>12×</sub> | 0.583<br><sub>686×</sub> | 0.616<br><sub>729×</sub> | 0.695<br><sub>806×</sub> |
| bank4 | 0.606% | 0.487<br><sub>137×</sub> | 0.773<br><sub>170×</sub> | 0.772<br><sub>168×</sub> | 0.788<br><sub>176×</sub> | 0.754<br><sub>169×</sub> | 0.762<br><sub>165×</sub> | 0.751<br><sub>157×</sub> | 0.779<br><sub>170×</sub> |

Pearson r between a cell's AUC-PR and its prevalence at n=500 (15 cells): fedavg **+0.79**, local-only +0.83, pooled +0.73. A strong positive r means most of the raw sd in that column is base rate.

### FedAvg vs coordinate-median with no attacker

The efficiency price of the robust rule, if any (AUC-PR difference, median − fedavg; paired wins for median):

- `iid`: n=50: +0.000 (7/15); n=500: -0.002 (4/15); n=5000: -0.001 (6/15); n=0: -0.001 (7/15)
- `amount`: n=50: -0.007 (2/15); n=500: -0.008 (3/15); n=5000: -0.011 (1/15); n=0: -0.006 (2/15)
- `cluster`: n=50: +0.018 (12/15); n=500: +0.014 (12/15); n=5000: +0.009 (10/15); n=0: +0.023 (13/15)

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

## Result 3 — threshold bridge: per-bank calibrated cutoffs vs one global cutoff

This is the mechanism the brief actually promises for Component B (§2.3). One FedAvg global model
is trained over all 5 banks per (split, seed) and used as the scorer everywhere; each bank's rows are
split 50/50 into calibration and evaluation halves; only the **cutoffs** differ between conditions.

- **Flag-rate targets** (unlabelled, day-one): ≥L1 5%, ≥L2 1%, ≥L3 0.1% of traffic.
  `global_flagrate` = one cutoff set from pooled calibration scores (the v1 "single number" analogue);
  `perbank_flagrate` = each bank's own.
- **Precision targets** (labelled, after `/feedback`): ≥L1 2%, ≥L2 10%, ≥L3 50%.
  `global_precision` from pooled labelled calibration data; `perbank_precision` from each bank's own.

Cells: 5 banks × 3 seeds = 15 per split. Fraud counts per bank-half are small
(tens), so ≥L3 precision/recall are noisy; read ≥L2 as the main operating point.

### `iid`

**Alert volume per bank at ≥L2 (target 1.00% of each bank's traffic)** — mean over seeds:

| Bank | eval prevalence | `global_flagrate` flag rate | `perbank_flagrate` flag rate | global recall | per-bank recall | global precision | per-bank precision |
|---|---|---|---|---|---|---|---|
| bank0 | 0.178% | 1.03% | 1.10% | 83.8% | 83.8% | 14.4% | 13.4% |
| bank1 | 0.189% | 1.02% | 0.99% | 91.3% | 91.3% | 17.0% | 17.4% |
| bank2 | 0.140% | 0.99% | 1.01% | 88.3% | 88.3% | 12.5% | 12.3% |
| bank3 | 0.191% | 1.04% | 1.00% | 86.6% | 86.6% | 16.0% | 16.6% |
| bank4 | 0.172% | 0.97% | 0.95% | 78.8% | 78.8% | 14.0% | 14.4% |

Across bank-cells the ≥L2 alert volume under the global cutoff ranges **0.94%–1.10%** (sd 0.04%) against a 1.00% budget; under per-bank calibration it ranges **0.85%–1.19%** (sd 0.08%). The per-bank residual is calibration-vs-evaluation sampling noise; the global spread is the distribution shift between banks.

**Totals over all banks, per level (same model, same overall alert budget):**

| Level | Strategy | total alerts / seed | total TP / seed | pooled precision | pooled recall |
|---|---|---|---|---|---|
| ≥L1 | `global_flagrate` | 7,143 | 220.7 | 3.1% | 89.1% |
| ≥L1 | `perbank_flagrate` | 7,165 | 220.7 | 3.1% | 89.1% |
| ≥L2 | `global_flagrate` | 1,437 | 212.7 | 14.8% | 85.9% |
| ≥L2 | `perbank_flagrate` | 1,439 | 212.7 | 14.8% | 85.9% |
| ≥L3 | `global_flagrate` | 135 | 112.0 | 82.5% | 45.3% |
| ≥L3 | `perbank_flagrate` | 141 | 117.0 | 83.1% | 47.3% |

**Precision targets — achieved ≥L2 precision per bank (target 10%), mean over seeds:**

| Bank | `global_precision` precision | `perbank_precision` precision | global recall | per-bank recall | global alerts | per-bank alerts |
|---|---|---|---|---|---|---|
| bank0 | 9.4% | 10.0% | 85.5% | 85.5% | 457 | 445 |
| bank1 | 11.1% | 11.8% | 91.3% | 91.3% | 443 | 424 |
| bank2 | 7.9% | 7.5% | 88.3% | 88.3% | 445 | 470 |
| bank3 | 10.8% | 11.1% | 87.3% | 87.3% | 440 | 429 |
| bank4 | 9.2% | 9.1% | 79.6% | 80.3% | 424 | 434 |

Bank-cells meeting the ≥L2 precision target on their evaluation half: global cutoff **7/15**, per-bank cutoff **8/15**.

**How different are the per-bank cutoffs?** (probability cutoff for ≥L2, mean over seeds; global cutoff for reference)

| | bank0 | bank1 | bank2 | bank3 | bank4 | global | max/min ratio |
|---|---|---|---|---|---|---|---|
| `perbank_flagrate` | 0.2770 | 0.2887 | 0.2847 | 0.2913 | 0.2882 | 0.2862 | 1.1× |
| `perbank_precision` | 0.2375 | 0.2391 | 0.2241 | 0.2363 | 0.2324 | 0.2332 | 1.1× |

### `amount`

**Alert volume per bank at ≥L2 (target 1.00% of each bank's traffic)** — mean over seeds:

| Bank | eval prevalence | `global_flagrate` flag rate | `perbank_flagrate` flag rate | global recall | per-bank recall | global precision | per-bank precision |
|---|---|---|---|---|---|---|---|
| bank0 | 0.291% | 3.10% | 1.04% | 89.3% | 87.2% | 8.3% | 24.5% |
| bank1 | 0.118% | 0.52% | 1.06% | 86.4% | 89.4% | 19.6% | 10.0% |
| bank2 | 0.107% | 0.46% | 1.07% | 83.8% | 83.8% | 19.5% | 8.4% |
| bank3 | 0.147% | 0.53% | 1.01% | 94.3% | 95.1% | 26.1% | 14.1% |
| bank4 | 0.211% | 0.66% | 1.09% | 79.5% | 80.6% | 25.4% | 15.6% |

Across bank-cells the ≥L2 alert volume under the global cutoff ranges **0.44%–3.19%** (sd 1.06%) against a 1.00% budget; under per-bank calibration it ranges **0.87%–1.18%** (sd 0.09%). The per-bank residual is calibration-vs-evaluation sampling noise; the global spread is the distribution shift between banks.

**Totals over all banks, per level (same model, same overall alert budget):**

| Level | Strategy | total alerts / seed | total TP / seed | pooled precision | pooled recall |
|---|---|---|---|---|---|
| ≥L1 | `global_flagrate` | 7,194 | 224.0 | 3.1% | 90.0% |
| ≥L1 | `perbank_flagrate` | 7,295 | 224.3 | 3.1% | 90.1% |
| ≥L2 | `global_flagrate` | 1,502 | 215.7 | 14.4% | 86.6% |
| ≥L2 | `perbank_flagrate` | 1,502 | 216.0 | 14.4% | 86.8% |
| ≥L3 | `global_flagrate` | 156 | 129.0 | 82.7% | 51.8% |
| ≥L3 | `perbank_flagrate` | 167 | 127.0 | 75.9% | 51.0% |

**Precision targets — achieved ≥L2 precision per bank (target 10%), mean over seeds:**

| Bank | `global_precision` precision | `perbank_precision` precision | global recall | per-bank recall | global alerts | per-bank alerts |
|---|---|---|---|---|---|---|
| bank0 | 5.7% | 9.2% | 92.1% | 89.3% | 1345 | 821 |
| bank1 | 13.3% | 12.3% | 87.3% | 89.4% | 222 | 246 |
| bank2 | 13.3% | 9.1% | 83.8% | 83.8% | 193 | 281 |
| bank3 | 19.3% | 10.8% | 95.1% | 95.1% | 207 | 372 |
| bank4 | 19.2% | 9.9% | 79.5% | 81.2% | 250 | 502 |

Bank-cells meeting the ≥L2 precision target on their evaluation half: global cutoff **12/15**, per-bank cutoff **7/15**.

**How different are the per-bank cutoffs?** (probability cutoff for ≥L2, mean over seeds; global cutoff for reference)

| | bank0 | bank1 | bank2 | bank3 | bank4 | global | max/min ratio |
|---|---|---|---|---|---|---|---|
| `perbank_flagrate` | 0.5012 | 0.2262 | 0.1920 | 0.2101 | 0.2227 | 0.3458 | 2.6× |
| `perbank_precision` | 0.3554 | 0.2577 | 0.2033 | 0.1694 | 0.1487 | 0.2740 | 2.4× |

### `cluster`

**Alert volume per bank at ≥L2 (target 1.00% of each bank's traffic)** — mean over seeds:

| Bank | eval prevalence | `global_flagrate` flag rate | `perbank_flagrate` flag rate | global recall | per-bank recall | global precision | per-bank precision |
|---|---|---|---|---|---|---|---|
| bank0 | 0.088% | 1.18% | 0.96% | 77.0% | 74.1% | 8.2% | 6.7% |
| bank1 | 0.157% | 1.16% | 1.03% | 73.7% | 75.0% | 10.0% | 11.5% |
| bank2 | 0.613% | 2.24% | 1.07% | 95.7% | 91.0% | 28.4% | 53.4% |
| bank3 | 0.066% | 0.49% | 1.01% | 73.8% | 75.4% | 10.3% | 4.9% |
| bank4 | 0.422% | 1.92% | 1.09% | 88.3% | 84.1% | 20.7% | 36.5% |

Across bank-cells the ≥L2 alert volume under the global cutoff ranges **0.36%–2.56%** (sd 0.82%) against a 1.00% budget; under per-bank calibration it ranges **0.93%–1.22%** (sd 0.08%). The per-bank residual is calibration-vs-evaluation sampling noise; the global spread is the distribution shift between banks.

**Totals over all banks, per level (same model, same overall alert budget):**

| Level | Strategy | total alerts / seed | total TP / seed | pooled precision | pooled recall |
|---|---|---|---|---|---|
| ≥L1 | `global_flagrate` | 7,192 | 223.0 | 3.1% | 90.8% |
| ≥L1 | `perbank_flagrate` | 7,200 | 223.3 | 3.1% | 90.9% |
| ≥L2 | `global_flagrate` | 1,437 | 212.7 | 14.8% | 86.5% |
| ≥L2 | `perbank_flagrate` | 1,453 | 207.7 | 14.3% | 84.4% |
| ≥L3 | `global_flagrate` | 154 | 114.0 | 73.7% | 46.2% |
| ≥L3 | `perbank_flagrate` | 166 | 99.0 | 59.5% | 40.1% |

**Precision targets — achieved ≥L2 precision per bank (target 10%), mean over seeds:**

| Bank | `global_precision` precision | `perbank_precision` precision | global recall | per-bank recall | global alerts | per-bank alerts |
|---|---|---|---|---|---|---|
| bank0 | 4.9% | 10.5% | 78.3% | 78.3% | 392 | 179 |
| bank1 | 6.8% | 9.0% | 75.4% | 74.3% | 569 | 410 |
| bank2 | 23.3% | 8.4% | 96.0% | 96.0% | 342 | 775 |
| bank3 | 6.1% | 9.9% | 75.7% | 73.8% | 399 | 233 |
| bank4 | 15.4% | 10.6% | 88.3% | 88.3% | 452 | 590 |

Bank-cells meeting the ≥L2 precision target on their evaluation half: global cutoff **3/15**, per-bank cutoff **6/15**.

**How different are the per-bank cutoffs?** (probability cutoff for ≥L2, mean over seeds; global cutoff for reference)

| | bank0 | bank1 | bank2 | bank3 | bank4 | global | max/min ratio |
|---|---|---|---|---|---|---|---|
| `perbank_flagrate` | 0.4516 | 0.3203 | 0.8942 | 0.2387 | 0.6775 | 0.3151 | 3.7× |
| `perbank_precision` | 0.4013 | 0.2991 | 0.1776 | 0.3111 | 0.2557 | 0.2546 | 2.3× |

### Reading Result 3

- **`iid`:** per-bank >=L2 cutoffs differ by **1.1x** across banks. A single global cutoff sends **0.94%-1.10%** of each bank's traffic to review against a 1% budget; per-bank calibration holds it at **0.85%-1.19%**. For the same total alert volume, pooled >=L2 recall changes by **+0.0 pts** (per-bank minus global). Labelled precision-target calibration meets its >=L2 target in 7/15 bank-cells with the global cutoff vs 8/15 per-bank.
- **`amount`:** per-bank >=L2 cutoffs differ by **2.6x** across banks. A single global cutoff sends **0.44%-3.19%** of each bank's traffic to review against a 1% budget; per-bank calibration holds it at **0.87%-1.18%**. For the same total alert volume, pooled >=L2 recall changes by **+0.1 pts** (per-bank minus global). Labelled precision-target calibration meets its >=L2 target in 12/15 bank-cells with the global cutoff vs 7/15 per-bank.
- **`cluster`:** per-bank >=L2 cutoffs differ by **3.7x** across banks. A single global cutoff sends **0.36%-2.56%** of each bank's traffic to review against a 1% budget; per-bank calibration holds it at **0.93%-1.22%**. For the same total alert volume, pooled >=L2 recall changes by **-2.1 pts** (per-bank minus global). Labelled precision-target calibration meets its >=L2 target in 3/15 bank-cells with the global cutoff vs 6/15 per-bank.

**What the bridge demonstrably buys, on this data:** *alert-budget adherence per bank.* Under the non-IID splits a
single cutoff over-alerts the bank whose score distribution sits highest and starves the others - a 1% budget becomes
several percent at one bank and a fraction of a percent at another. Per-bank flag-rate calibration removes that, with no
labels needed.

**What it does not buy here:** *more fraud caught.* For the same total number of alerts, pooled >=L2 recall under per-bank
budgets minus recall under the global cutoff is +0.0 pts (`iid`), +0.1 pts (`amount`), -2.1 pts (`cluster`). Per-bank budgets
never gain more than +0.1 pts and give up 2.1 pts under `cluster`. That is expected from the construction: the global model's
probabilities are comparable across banks, so a single cutoff already allocates alerts to the highest-scoring rows regardless
of bank - the recall-optimal allocation of a fixed total budget. Per-bank budgets trade some of that optimality for predictable
ops load at every bank. Which one a federation wants is a policy decision, not a modelling one, and the bridge supports both.

**Labelled per-bank precision calibration is unreliable at this fraud volume.** Across all splits the >=L2 precision target
is met on the evaluation half in 22/45 bank-cells with the global cutoff and 21/45 with per-bank cutoffs, and
which strategy does better flips by split. With tens of confirmed frauds per bank-half, a cutoff fitted to one half does not
hold on the other whichever way it is fitted. Precision-target calibration should wait until `/feedback` has accumulated
hundreds of confirmed outcomes per bank; until then the bridge should run on flag-rate calibration. Under `iid` the flag-rate
strategies are, as expected, indistinguishable - that row is the control.

## Limitations

1. Card fraud, not UPI social engineering; PCA features, not Kavach's contract. Mechanism result only.
2. "Banks" are synthetic partitions of one dataset. Real banks differ in ways a mixing matrix does not capture
   (different fraud typologies, different labelling latency and quality, different base rates by an order of magnitude).
3. Logistic regression is a deliberately simple learner. The federated-vs-local *gap* is what is being measured, and a
   stronger local learner would shrink it at large history — the small-history end of the curve is the claim, not the right end.
4. Attackers here are crude (sign flip, noise, inflated counts). A stealthy attacker who shifts deltas by a small,
   consistent amount every round is not tested and is *not* defeated by a median.
5. No differential privacy or secure aggregation: the aggregator sees each bank's delta in the clear. "No raw data leaves
   the bank" is true; "nothing about the bank's data can be inferred from its delta" is not claimed.
6. Fine-tune epochs, rounds and learning rate were set once, not tuned per method. Tuning would move numbers, not the shape.
7. Result 3 uses the same global model at every bank so that only the cutoffs differ; in deployment each bank also
   fine-tunes, which would change the score distributions the cutoffs are calibrated on.

_Run time 3.6 min. Raw trials in `cold_start_trials.json`._
