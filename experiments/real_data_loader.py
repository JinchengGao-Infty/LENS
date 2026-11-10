"""Schema-declared loaders for the real graph-utility datasets.

HELOC (Home Equity Line of Credit):
  - Source: IBM AIX360 / FICO community
  - ~10k samples, 23 features, binary target (RiskPerformance)
  - Special values: -7 (condition not met), -8 (no usable/valid data),
    -9 (no bureau record or trade) → treated as missing

UCI Default of Credit Card Clients:
  - Source: UCI ML Repository
  - 30k samples, 23 features, binary target (default.payment.next.month)
  - Features include 6-month delinquency, bill, payment history
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HELOC
# ---------------------------------------------------------------------------

HELOC_SPECIAL_VALUES = {-7, -8, -9}

# Features that are conceptually "count" or "months" — special values here
# mean "no trade / condition not met", reasonable to impute with 0 or a
# dedicated indicator.
HELOC_FEATURE_NAMES = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DeijWorst",
    "NumTrades90Ever2DeijWorst",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]


def load_heloc(data_dir: str | Path = "data") -> dict:
    """Load and preprocess HELOC dataset.

    Preprocessing:
      1. Replace special values (-7, -8, -9) with NaN
      2. Impute NaN with column median (robust to skew)
      3. Binary target: 1 = Bad (delinquent), 0 = Good

    Returns dict with keys: X, y, feature_names, n, p, description
    """
    data_dir = Path(data_dir)
    fpath = data_dir / "heloc_dataset_v1.csv"
    if not fpath.exists():
        raise FileNotFoundError(
            f"HELOC dataset not found at {fpath}. "
            "Download from https://community.fico.com/s/explainable-machine-learning-challenge "
            "and place heloc_dataset_v1.csv in the data/ directory."
        )

    df = pd.read_csv(fpath)

    # Target: RiskPerformance
    y = (df["RiskPerformance"] == "Bad").astype(int).values

    # Features
    feat_cols = [c for c in df.columns if c != "RiskPerformance"]
    X = df[feat_cols].values.astype(np.float64)

    # Replace special values with NaN.
    # XGBoost handles NaN natively (learns optimal default direction per split),
    # so NO global imputation — avoids cross-fold data leakage.
    # Downstream sklearn models must impute per-fold (see real_data_analysis.py).
    for sv in HELOC_SPECIAL_VALUES:
        X[X == sv] = np.nan

    n_missing = np.isnan(X).sum()
    pct_missing = n_missing / X.size * 100
    print(f"  HELOC: {n_missing} missing values ({pct_missing:.1f}%) after special-value removal")

    return {
        "X": X,
        "y": y,
        "feature_names": feat_cols,
        "feature_kinds": feature_kinds_for_names("heloc", feat_cols),
        "n": X.shape[0],
        "p": X.shape[1],
        "task": "binary",
        "description": "HELOC (Home Equity Line of Credit)",
    }


# ---------------------------------------------------------------------------
# UCI Credit Card
# ---------------------------------------------------------------------------

UCI_CREDIT_FEATURE_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

UCI_CREDIT_CATEGORICAL = {
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
}
HELOC_CATEGORICAL = {"MaxDelq2PublicRecLast12M", "MaxDelqEver"}
FREMTPL2_CATEGORICAL = {"VehBrand", "VehGas", "Region", "Area"}
BIKE_SHARING_CATEGORICAL = {
    "season",
    "year",
    "month",
    "hour",
    "holiday",
    "weekday",
    "workingday",
    "weather",
}


def feature_kinds_for_names(dataset: str, feature_names: list[str]) -> list[str]:
    """Return schema-declared kinds; never infer semantic type from cardinality."""
    categorical_by_dataset = {
        "heloc": HELOC_CATEGORICAL,
        "uci_credit": UCI_CREDIT_CATEGORICAL,
        "fremtpl2": FREMTPL2_CATEGORICAL,
        "california_housing": set(),
        "bike_sharing": BIKE_SHARING_CATEGORICAL,
    }
    try:
        categorical = categorical_by_dataset[dataset]
    except KeyError as exc:
        raise ValueError(f"no feature-kind schema registered for {dataset!r}") from exc
    return ["categorical" if name in categorical else "continuous" for name in feature_names]


def load_uci_credit(data_dir: str | Path = "data") -> dict:
    """Load and preprocess UCI Default of Credit Card Clients dataset.

    The dataset has a header row at row 1 (0-indexed) in the .xls file.
    Feature columns: X2–X24 in the original; we rename to meaningful names.

    Returns dict with keys: X, y, feature_names, n, p, description
    """
    data_dir = Path(data_dir)

    # Try multiple common file names
    candidates = [
        "default of credit card clients.xls",
        "default_of_credit_card_clients.xls",
        "UCI_Credit_Card.csv",
        "default of credit card clients.csv",
    ]
    fpath = None
    for name in candidates:
        p = data_dir / name
        if p.exists():
            fpath = p
            break

    if fpath is None:
        raise FileNotFoundError(
            f"UCI Credit Card dataset not found in {data_dir}. "
            "Download from https://archive.ics.uci.edu/dataset/350/ "
            "and place the file in the data/ directory."
        )

    if fpath.suffix == ".xls":
        # Original .xls format: skip first row (header), use second row as header
        df = pd.read_excel(fpath, header=1)
    else:
        df = pd.read_csv(fpath)

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Target column (various names used in different versions)
    target_col = None
    for tc in ["default payment next month", "default.payment.next.month", "Y"]:
        if tc in df.columns:
            target_col = tc
            break
    if target_col is None:
        raise ValueError(f"Cannot find target column. Available: {list(df.columns)}")

    y = df[target_col].astype(int).values

    # Features: everything except target
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].values.astype(np.float64)

    # Basic cleaning: PAY_0 etc. can have -2, -1 (pay duly / no consumption)
    # These are valid ordinal values, keep as-is.
    # No missing values in this dataset typically.
    assert not np.isnan(X).any(), "Unexpected NaN in UCI Credit Card data"

    return {
        "X": X,
        "y": y,
        "feature_names": feat_cols,
        "feature_kinds": feature_kinds_for_names("uci_credit", feat_cols),
        "n": X.shape[0],
        "p": X.shape[1],
        "task": "binary",
        "description": "UCI Default of Credit Card Clients",
    }


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# freMTPL2 (French Motor Third-Party Liability Insurance)
# ---------------------------------------------------------------------------

def load_fremtpl2(data_dir: str | Path = "data") -> dict:
    """Load and preprocess freMTPL2 insurance dataset.

    Uses sklearn's OpenML interface (data_id=41214).
    Target: binary indicator of whether the policy had at least one claim
    (ClaimNb > 0). This is the standard binary-classification formulation
    used in insurance pricing benchmarks.

    Features: VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas,
    Area, Density, Region. We drop IDpol (policy ID), Exposure (offset),
    and ClaimNb (target source).

    Subsample to 50k for tractability (full dataset is 678k).
    """
    from sklearn.datasets import fetch_openml

    data_dir = Path(data_dir)
    cache_path = data_dir / "fremtpl2_processed.npz"

    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        return {
            "X": loaded["X"],
            "y": loaded["y"],
            "feature_names": loaded["feature_names"].tolist(),
            "feature_kinds": feature_kinds_for_names(
                "fremtpl2", loaded["feature_names"].tolist()
            ),
            "n": int(loaded["n"]),
            "p": int(loaded["p"]),
            "task": "binary",
            "description": "freMTPL2 (French Motor Insurance)",
        }

    print("  Downloading freMTPL2 from OpenML (first time only)...")
    bunch = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    df = bunch.data

    # Target: at least one claim
    y = (df["ClaimNb"].astype(float) > 0).astype(int).values

    # Drop non-feature columns
    drop_cols = ["IDpol", "ClaimNb", "Exposure"]
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    for col in feat_df.select_dtypes(include=["category", "object"]).columns:
        feat_df[col] = feat_df[col].astype("category").cat.codes.astype(float)

    feat_cols = feat_df.columns.tolist()
    X = feat_df.values.astype(np.float64)

    # Subsample to 50k for tractability
    rng = np.random.RandomState(42)
    n_sub = min(50000, len(X))
    idx = rng.choice(len(X), n_sub, replace=False)
    X = X[idx]
    y = y[idx]

    # Cache
    np.savez(
        cache_path, X=X, y=y,
        feature_names=np.array(feat_cols),
        feature_kinds=np.array(feature_kinds_for_names("fremtpl2", feat_cols)),
        n=X.shape[0], p=X.shape[1],
    )

    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"  freMTPL2: {n_missing} missing values")

    return {
        "X": X,
        "y": y,
        "feature_names": feat_cols,
        "feature_kinds": feature_kinds_for_names("fremtpl2", feat_cols),
        "n": X.shape[0],
        "p": X.shape[1],
        "task": "binary",
        "description": "freMTPL2 (French Motor Insurance)",
    }


def load_california_housing(data_dir: str | Path = "data") -> dict:
    """Load sklearn's canonical California Housing regression dataset."""
    from sklearn.datasets import fetch_california_housing

    data_dir = Path(data_dir)
    cache_path = data_dir / "california_housing_processed.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        feature_names = loaded["feature_names"].tolist()
        return {
            "X": loaded["X"],
            "y": loaded["y"],
            "feature_names": feature_names,
            "feature_kinds": feature_kinds_for_names(
                "california_housing", feature_names
            ),
            "n": int(loaded["n"]),
            "p": int(loaded["p"]),
            "task": "regression",
            "description": "California Housing",
        }

    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_california_housing(
        data_home=data_dir / "sklearn_cache",
        as_frame=True,
    )
    feature_names = list(bunch.feature_names)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("California Housing contains unexpected non-finite values")
    np.savez(
        cache_path,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names),
        n=X.shape[0],
        p=X.shape[1],
    )
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "feature_kinds": feature_kinds_for_names(
            "california_housing", feature_names
        ),
        "n": X.shape[0],
        "p": X.shape[1],
        "task": "regression",
        "description": "California Housing",
    }


def load_bike_sharing(data_dir: str | Path = "data") -> dict:
    """Load OpenML 42712 Bike Sharing Demand with a fixed public schema.

    The OpenML target is hourly rental ``count``. The source already excludes
    the target components ``casual`` and ``registered``, so neither can leak
    into the feature matrix.
    """
    from sklearn.datasets import fetch_openml

    data_dir = Path(data_dir)
    cache_path = data_dir / "bike_sharing_processed.npz"
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        feature_names = loaded["feature_names"].tolist()
        return {
            "X": loaded["X"],
            "y": loaded["y"],
            "feature_names": feature_names,
            "feature_kinds": feature_kinds_for_names("bike_sharing", feature_names),
            "n": int(loaded["n"]),
            "p": int(loaded["p"]),
            "task": "regression",
            "description": "Bike Sharing Demand (OpenML 42712)",
        }

    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        data_id=42712,
        as_frame=True,
        parser="auto",
        data_home=data_dir / "openml_cache",
    )
    frame = bunch.data.copy()
    expected_columns = [
        "season",
        "year",
        "month",
        "hour",
        "holiday",
        "weekday",
        "workingday",
        "weather",
        "temp",
        "feel_temp",
        "humidity",
        "windspeed",
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "OpenML 42712 schema changed: "
            f"expected {expected_columns}, got {list(frame.columns)}"
        )
    fixed_mappings = {
        "season": {"spring": 1.0, "summer": 2.0, "fall": 3.0, "winter": 4.0},
        "holiday": {"False": 0.0, "True": 1.0},
        "workingday": {"False": 0.0, "True": 1.0},
        "weather": {
            "clear": 1.0,
            "misty": 2.0,
            "rain": 3.0,
            "heavy_rain": 4.0,
        },
    }
    for column, mapping in fixed_mappings.items():
        observed = set(frame[column].astype(str))
        if not observed.issubset(mapping):
            raise ValueError(
                f"OpenML 42712 has unknown {column} levels: {sorted(observed - set(mapping))}"
            )
        frame[column] = frame[column].astype(str).map(mapping)
    feature_names = expected_columns
    X = frame.to_numpy(dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Bike Sharing contains unexpected non-finite values")
    np.savez(
        cache_path,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names),
        n=X.shape[0],
        p=X.shape[1],
    )
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "feature_kinds": feature_kinds_for_names("bike_sharing", feature_names),
        "n": X.shape[0],
        "p": X.shape[1],
        "task": "regression",
        "description": "Bike Sharing Demand (OpenML 42712)",
    }


DATASETS = {
    "heloc": load_heloc,
    "uci_credit": load_uci_credit,
    "fremtpl2": load_fremtpl2,
    "california_housing": load_california_housing,
    "bike_sharing": load_bike_sharing,
}

DATASET_TASKS = {
    "heloc": "binary",
    "uci_credit": "binary",
    "fremtpl2": "binary",
    "california_housing": "regression",
    "bike_sharing": "regression",
}


def get_dataset_task(name: str) -> str:
    try:
        return DATASET_TASKS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {list(DATASETS.keys())}"
        ) from exc


def load_dataset(name: str, data_dir: str | Path = "data") -> dict:
    """Load a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](data_dir=data_dir)
