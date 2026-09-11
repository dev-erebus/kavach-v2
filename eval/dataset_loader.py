"""
Dataset loading for the evaluation harness.

Supported sources
-----------------
* ``ulb`` — the ULB / Kaggle "Credit Card Fraud Detection" dataset (Dal Pozzolo
  et al., 2015). Fetched from OpenML (dataset id 1597) which needs no Kaggle
  credentials, or from a local ``creditcard.csv`` if you already downloaded it.
* ``ieee_cis`` — IEEE-CIS Fraud Detection (Kaggle competition). Kaggle-only, so
  it must be downloaded manually; pass the directory containing
  ``train_transaction.csv``. Loader is provided; the main harness run uses ULB.

READ THIS BEFORE INTERPRETING ANY NUMBER DOWNSTREAM
----------------------------------------------------
Both datasets are **card-present / e-commerce card fraud in Europe / North
America**. They are **not** UPI/IMPS social-engineering fraud in India. They
contain no phone-call context, no sender age, no beneficiary-account age, no
device location — none of the signals Kavach's L2/L3 logic is built around.

To push these rows through the *real* Kavach pipeline (adapter → canonical
schema → classifier) rather than a side-channel, ``to_kavach_transactions``
fills the missing canonical fields with fixed **neutral** values that cannot
trigger any social-engineering rule: sender age 40, receiver account 10 years
old, no active call, biometric unknown. This is deliberate. It means the v1
rule pipeline degrades to an amount-only classifier on this data, and the
report says so. The alternative — synthesising fraud-correlated signals — would
be manufacturing evidence, so it is not done.

Amounts are in EUR. They are placed on the INR rule scale with a single fixed
constant (``eur_to_inr``) purely so the amount thresholds are exercised. The
constant is documented in the report and does not affect rank-based metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import pandas as pd

from adapters.base_adapter import BankAdapter, NormalizationError
from adapters.schema import KavachTransaction, PaymentRail

DatasetName = Literal["ulb", "ieee_cis"]


@dataclass
class LabeledDataset:
    name: str
    provenance: str
    frame: pd.DataFrame  # feature columns + 'amount' + 'label'
    feature_columns: list[str]  # the dataset's OWN features (V1..V28 etc.) — for reference baselines
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def n_fraud(self) -> int:
        return int(self.frame["label"].sum())

    @property
    def prevalence(self) -> float:
        return self.n_fraud / self.n if self.n else float("nan")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_ulb(local_csv: Optional[Path] = None, cache_dir: Path = Path("data/openml_cache")) -> LabeledDataset:
    """ULB credit-card dataset. Prefers a local Kaggle CSV if given, else OpenML 1597."""
    notes: list[str] = []
    if local_csv is not None:
        df = pd.read_csv(local_csv)
        provenance = f"local file {local_csv} (Kaggle 'Credit Card Fraud Detection', mlg-ulb)"
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])
            notes.append("Dropped 'Time' (seconds since first transaction); not used by any Kavach rule.")
    else:
        from sklearn.datasets import fetch_openml

        t0 = time.time()
        ds = fetch_openml("creditcard", version=1, as_frame=True, data_home=str(cache_dir), parser="auto")
        df = ds.frame
        provenance = (
            "OpenML dataset id 1597 'creditcard' v1 (mirror of the ULB/Kaggle dataset; "
            f"fetched in {time.time() - t0:.0f}s; OpenML's copy omits the 'Time' column)"
        )
    df = df.rename(columns={"Amount": "amount", "Class": "label"})
    df["label"] = df["label"].astype(int)
    df["amount"] = df["amount"].astype(float)
    feature_cols = [c for c in df.columns if c.startswith("V")]
    notes.append("Amounts are in EUR (European cardholders, Sept 2013). Features V1..V28 are anonymised PCA components.")
    return LabeledDataset("ulb", provenance, df.reset_index(drop=True), feature_cols, notes)


def load_ieee_cis(directory: Path) -> LabeledDataset:
    """IEEE-CIS Fraud Detection (Kaggle). Needs ``train_transaction.csv`` downloaded manually."""
    path = Path(directory) / "train_transaction.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. IEEE-CIS is Kaggle-only; download it from "
            "https://www.kaggle.com/c/ieee-fraud-detection/data and pass the directory."
        )
    df = pd.read_csv(path)
    df = df.rename(columns={"TransactionAmt": "amount", "isFraud": "label"})
    # Keep numeric columns only for a reference baseline; the harness never uses them for rules.
    numeric = [c for c in df.columns if c.startswith(("C", "D", "V")) and pd.api.types.is_numeric_dtype(df[c])]
    df = df[["amount", "label"] + numeric].fillna(0.0)
    return LabeledDataset(
        "ieee_cis",
        f"local Kaggle download at {path}",
        df.reset_index(drop=True),
        numeric,
        ["Amounts are in USD. e-commerce card-not-present fraud; no social-engineering context fields."],
    )


def load(name: DatasetName, path: Optional[Path] = None) -> LabeledDataset:
    if name == "ulb":
        return load_ulb(local_csv=path)
    if name == "ieee_cis":
        if path is None:
            raise ValueError("ieee_cis requires --path <dir containing train_transaction.csv>")
        return load_ieee_cis(path)
    raise ValueError(f"unknown dataset '{name}'")


# --------------------------------------------------------------------------- #
# Mapping into the real pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalMappingConfig:
    """How dataset rows become canonical transactions. Every value here is a stated assumption."""

    #: Places EUR/USD amounts on the INR rule scale. Rank metrics don't depend on it.
    fx_to_inr: Decimal = Decimal("90")
    #: Neutral fills — chosen so NO social-engineering rule can fire from them.
    neutral_sender_age: int = 40
    neutral_receiver_account_age_days: int = 3650
    neutral_is_active_phone_call: bool = False
    bank_id: str = "eval_dataset"


class EvalDatasetAdapter(BankAdapter):
    """Maps one dataset row into ``KavachTransaction`` via the same ``_build`` path real adapters use.

    Using a ``BankAdapter`` here is not ceremony: it means the eval rows are
    subject to exactly the canonical validation the LLM layer relies on
    (brief §1.4 criterion 2), and any row the schema rejects is counted and
    reported instead of vanishing.
    """

    core_banking_system = "evaluation dataset (not a bank)"

    def __init__(self, cfg: EvalMappingConfig):
        self.cfg = cfg
        self.bank_id = cfg.bank_id
        self._epoch = datetime(2013, 9, 1, tzinfo=timezone.utc)  # ULB collection month; cosmetic

    def validate_auth(self, headers: Mapping[str, str], body: bytes = b"") -> bool:
        return True  # offline batch source; nothing to authenticate

    def normalize(self, raw_payload: Mapping[str, Any]) -> KavachTransaction:
        c = self.cfg
        amount = (Decimal(str(raw_payload["amount"])) * c.fx_to_inr).quantize(Decimal("0.01"))
        return self._build(
            {
                "transaction_id": f"row-{raw_payload['row']}",
                "bank_id": self.bank_id,
                "timestamp_utc": self._epoch,
                "amount_inr": amount,
                "rail": PaymentRail.CARD,
                "sender_age": c.neutral_sender_age,
                "sender_account_age_days": None,
                "receiver_account_age_days": c.neutral_receiver_account_age_days,
                "receiver_name": None,
                "is_new_beneficiary": None,
                "is_active_phone_call": c.neutral_is_active_phone_call,
                "caller_verified_biometric": None,  # unknown, never coerced
                "device_location": "unknown (dataset has no device location)",
                "remarks": None,
            }
        )


@dataclass
class MappedDataset:
    transactions: list[KavachTransaction]
    labels: np.ndarray  # aligned with transactions
    kept_index: np.ndarray  # row indices in the source frame that survived validation
    rejected: dict[str, int]  # reason → count
    rejected_fraud: int  # how many of the rejected rows were labelled fraud — must be reported, not lost
    config: EvalMappingConfig


def to_kavach_transactions(ds: LabeledDataset, cfg: EvalMappingConfig | None = None) -> MappedDataset:
    """Push every row through the adapter → schema path. Rejections are counted, never dropped silently."""
    cfg = cfg or EvalMappingConfig()
    adapter = EvalDatasetAdapter(cfg)
    txs: list[KavachTransaction] = []
    kept: list[int] = []
    rejected: dict[str, int] = {}
    rejected_fraud = 0
    amounts = ds.frame["amount"].to_numpy()
    labels = ds.frame["label"].to_numpy()
    for i in range(len(ds.frame)):
        try:
            txs.append(adapter.normalize({"row": i, "amount": amounts[i]}))
            kept.append(i)
        except NormalizationError as e:
            key = f"{e.code}: {e.message}"
            rejected[key] = rejected.get(key, 0) + 1
            rejected_fraud += int(labels[i])
    kept_idx = np.asarray(kept, dtype=int)
    return MappedDataset(txs, labels[kept_idx], kept_idx, rejected, rejected_fraud, cfg)
