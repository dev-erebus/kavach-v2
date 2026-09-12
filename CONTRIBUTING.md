# Contributing to Kavach v2

Short version: run the tests, read the caveats, and pick one of the three contributions that would
actually move this project. Everything else is welcome but secondary.

## Where to start

```bash
git clone <this-repo> kavach-v2 && cd kavach-v2
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                         # pinned; see requirements.txt header
pytest -q                                               # 145 tests, offline, ~5 s
```

Then read, in this order: [`README.md`](README.md) → [`RESULTS.md`](RESULTS.md) → the build brief
([`kavach-v2-build-brief.md`](kavach-v2-build-brief.md)) → the module docstring of whatever you plan to
touch. Every module opens with a docstring that states what it does, what it assumes, and what it does
*not* claim. Keep that convention.

## The three contributions that matter

### 1. A domain-appropriate labelled dataset (Tier 2 on real data)

Every result in this repo is on European card-present fraud. The single most valuable contribution is
running the same harness on **labelled UPI/IMPS social-engineering fraud** — anything with confirmed
outcomes and, ideally, the canonical fields (`is_active_phone_call`, `sender_age`,
`receiver_account_age_days`). You do not need to share the data: the harness runs locally and produces
markdown reports; sharing the *report* is the contribution.

- Loader: subclass or extend `eval/dataset_loader.py::load()` to return a `LabeledDataset` with an
  `amount` and `label` column and whatever feature columns you have.
- Mapping: write a `BankAdapter` that fills `KavachTransaction` from your rows (see
  `EvalDatasetAdapter` for the pattern). Do **not** fill missing fields with fraud-correlated guesses —
  fill them neutrally and say so, as the ULB run does.
- Run `python -m eval.run_eval --dataset <yours>` and, if you have ≥3 natural partitions, the federated
  experiments. Open an issue with the "I have data" template and attach the reports.

### 2. Adapter mappings against a real core-banking API contract

`adapters/finacle_adapter.py`, `temenos_adapter.py` and `bancs_adapter.py` are *illustrative* — modelled
on publicly visible conventions, never checked against a vendor document. If you have access to a real
Finacle / T24 / BaNCS webhook specification (or any other core system), a mapping written against it is
worth more than all three current adapters combined.

- One file per adapter, one registry line; Nodes 3–10 untouched (`adapters/registry.py`).
- Every required field via `Section.require()` so rejections carry the full dotted path.
- Tri-state fields stay tri-state: `caller_verified_biometric` is `None` for unknown, never coerced.
- Add a test file mirroring `tests/test_registry_and_adapters.py`: happy path, one missing field, one
  bad value, auth valid / invalid / not configured.
- State in the module docstring exactly which document and version the mapping was verified against.

### 3. Stealthy poisoning attacks against the aggregator

The poisoning result (`federated/experiments/cold_start.py::poisoning`) uses crude attackers — sign flip,
noise, inflated sample counts — that the coordinate-median defeats. A small, consistent bias applied every
round is **not** tested and a median does **not** defeat it. New attack functions plug into `make_poison`;
report AUC-PR on the held-out bank at n=0 and n=5000, FedAvg vs coordinate-median vs trimmed-mean,
exactly as the existing table does. Attacks that beat the median are the interesting result.

## Rules that apply to every change

- **No claim without a table.** If a README or report sentence states a number, that number must appear
  in a generated table in `eval/output/` or `federated/output/`. If you change an experiment, re-run it and
  regenerate; do not hand-edit results.
- **Caveats are load-bearing.** Do not soften or remove a limitation to make a result read better. If
  you find a caveat is *wrong*, fix it and say why in the commit message.
- **Degenerate baselines are labelled, not headlined.** A comparison against a model that cannot do the
  task (no positive examples, constant scorer) is reported as a reference, never as the result.
- **Fail closed.** Missing secret → reject. Unknown spec → reject. Broken bridge → L2, never L0.
- **Tests offline.** `pytest` must pass with no network. Dataset-dependent checks live in the experiment
  scripts, not the test suite.
- Pin any new dependency in `requirements.txt` at the exact version you tested with.
- Commit in logical units with messages that say what changed and why.

## What is not wanted

- An in-house voice-liveness or deepfake detector. The interface is deliberate; a fabricated detector
  would emit confident verdicts with no basis.
- Filling in the RBI FR-2 field structure from memory. It needs a compliance reviewer with the current
  Master Direction in hand; until then every field stays `verified=False`.
- Synthetic fraud scenarios presented as evaluation. A payment simulator is on the roadmap as an
  *integration demo*; authored scenarios are not evidence of detection quality.
