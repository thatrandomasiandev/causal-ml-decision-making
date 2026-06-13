"""Real-world and semi-synthetic benchmark dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_ml.data.base import UpliftDataset
from causal_ml.data.paths import bundled_dir, data_dir, ensure_data_dir

IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"
IHDP_TRAIN_FALLBACK = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
)
JOBS_URL = (
    "https://raw.githubusercontent.com/uber/causalml/master/causalml/datasets/data/lalonde.csv"
)
JOBS_FALLBACK = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/sbw/lalonde.csv"
)
TWINS_BASE_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/"
)
TWINS_FILES = {
    "T": "twin_pairs_T_3years_samesex.csv",
    "X": "twin_pairs_X_3years_samesex.csv",
    "Y": "twin_pairs_Y_3years_samesex.csv",
}


@dataclass
class RealWorldDataset(UpliftDataset):
    """
    Real-world benchmark dataset with optional ground-truth CATE.

    Attributes:
        tau_true: Individual treatment effects when available, else None.
    """

    tau_true: np.ndarray | None = None
    overlap_diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau_true is not None:
            self.ground_truth = {**self.ground_truth, "tau": self.tau_true}


def _download(url: str, dest: Path, fallback_urls: list[str] | None = None) -> None:
    """Download a file if not present, trying fallback URLs on failure."""
    if dest.exists():
        return

    import requests

    urls = [url, *(fallback_urls or [])]
    errors: list[str] = []
    for candidate in urls:
        try:
            response = requests.get(candidate, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            errors.append(f"{candidate}: {exc}")

    raise FileNotFoundError(
        f"Failed to download data to {dest}. "
        f"Tried {len(urls)} URL(s). Place the file manually or check your network. "
        f"Errors: {'; '.join(errors)}"
    )


def _copy_bundled(filename: str, dest: Path) -> None:
    """Copy a bundled fallback dataset into data/raw."""
    source = bundled_dir() / filename
    if not source.exists():
        raise FileNotFoundError(
            f"Bundled fallback not found at {source}. "
            f"Download failed and no local copy is available."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())


def compute_overlap_diagnostics(X: np.ndarray, T: np.ndarray) -> dict[str, Any]:
    """
    Fit logistic propensity model and compute overlap diagnostics.

    Returns propensity distribution, effective sample size (ESS), and overlap flag.
    """
    T = T.astype(float).ravel()
    n = len(T)
    if len(np.unique(T)) < 2:
        return {
            "propensity": np.full(n, 0.5),
            "propensity_mean": 0.5,
            "propensity_std": 0.0,
            "propensity_min": 0.5,
            "propensity_max": 0.5,
            "ess": float(n),
            "ess_ratio": 1.0,
            "low_overlap": False,
            "note": "single treatment arm; propensity not estimated",
        }
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    model.fit(X, T)
    e = np.clip(model.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
    weights = T / e + (1 - T) / (1 - e)
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    ess_ratio = ess / n
    return {
        "propensity": e,
        "propensity_mean": float(np.mean(e)),
        "propensity_std": float(np.std(e)),
        "propensity_min": float(np.min(e)),
        "propensity_max": float(np.max(e)),
        "ess": ess,
        "ess_ratio": ess_ratio,
        "low_overlap": ess_ratio < 0.10,
    }


def _load_ihdp_csv(path: Path, realization: int, split: str) -> RealWorldDataset:
    """Load IHDP from causalml/CEVAE CSV format (single realization)."""
    df = pd.read_csv(path, header=None)
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [
        f"x{i}" for i in range(1, 26)
    ]
    df.columns = cols[: df.shape[1]]
    X = df[[c for c in df.columns if c.startswith("x")]].values.astype(float)
    T = df["treatment"].values.astype(float)
    Y = df["y_factual"].values.astype(float)
    mu0 = df["mu0"].values.astype(float)
    mu1 = df["mu1"].values.astype(float)
    tau_true = mu1 - mu0
    overlap = compute_overlap_diagnostics(X, T)
    return RealWorldDataset(
        X=X,
        T=T,
        Y=Y,
        tau_true=tau_true,
        overlap_diagnostics=overlap,
        metadata={
            "name": "ihdp",
            "split": split,
            "realization": realization,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "source": "csv",
        },
        ground_truth={"mu0": mu0, "mu1": mu1, "ate": float(np.mean(tau_true))},
    )


def load_ihdp(
    split: str = "train",
    realization: int = 1,
    download_if_missing: bool = True,
) -> RealWorldDataset:
    """
    Load IHDP semi-synthetic dataset (747 units, 25 covariates).

    Source: CFRNet IHDP benchmark (Hill 2011).
    """
    raw_dir = ensure_data_dir() if download_if_missing else data_dir()
    filename = f"ihdp_npci_1-100.{split}.npz"
    path = raw_dir / filename

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"IHDP data not found at {path}. "
                "Run with download_if_missing=True or place the file manually."
            )
        url = IHDP_TRAIN_URL if split == "train" else IHDP_TEST_URL
        csv_fallback = raw_dir / "ihdp_npci_1.csv"
        try:
            _download(url, path, fallback_urls=[IHDP_TRAIN_FALLBACK] if split == "train" else [])
        except FileNotFoundError:
            if split == "train" and not csv_fallback.exists():
                _download(IHDP_TRAIN_FALLBACK, csv_fallback)
            if csv_fallback.exists() and not path.exists():
                return _load_ihdp_csv(csv_fallback, realization, split)

    if path.suffix == ".csv" or (not path.exists() and (raw_dir / "ihdp_npci_1.csv").exists()):
        return _load_ihdp_csv(raw_dir / "ihdp_npci_1.csv", realization, split)

    data = np.load(path, allow_pickle=True)
    idx = realization - 1

    def _slice_x() -> np.ndarray:
        arr = np.asarray(data["x"])
        if arr.ndim != 3:
            return arr
        if arr.shape[2] <= 100 and arr.shape[2] < arr.shape[0]:
            return arr[:, :, min(idx, arr.shape[2] - 1)]
        if arr.shape[0] <= 100 and arr.shape[0] < arr.shape[1]:
            return arr[min(idx, arr.shape[0] - 1)]
        return arr[:, :, min(idx, arr.shape[2] - 1)]

    def _slice_vector(key: str) -> np.ndarray:
        arr = np.asarray(data[key])
        if arr.ndim == 2:
            if arr.shape[1] <= 100 and arr.shape[1] < arr.shape[0]:
                # (n, reps)
                return arr[:, min(idx, arr.shape[1] - 1)]
            if arr.shape[0] <= 100 and arr.shape[0] < arr.shape[1]:
                # (reps, n)
                return arr[min(idx, arr.shape[0] - 1)]
        return arr.ravel()

    X = _slice_x()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = _slice_vector("t").astype(float).ravel()
    Y = _slice_vector("yf").astype(float).ravel()
    mu0 = _slice_vector("mu0").astype(float).ravel()
    mu1 = _slice_vector("mu1").astype(float).ravel()
    tau_true = mu1 - mu0

    overlap = compute_overlap_diagnostics(X, T)

    return RealWorldDataset(
        X=X.astype(float),
        T=T,
        Y=Y,
        tau_true=tau_true,
        overlap_diagnostics=overlap,
        metadata={
            "name": "ihdp",
            "split": split,
            "realization": realization,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        },
        ground_truth={"mu0": mu0, "mu1": mu1, "ate": float(np.mean(tau_true))},
    )


def load_jobs(download_if_missing: bool = True) -> RealWorldDataset:
    """
    Load LaLonde Jobs training dataset (binary treatment and outcome).

    Ground-truth CATE is unavailable (tau_true=None).
    """
    raw_dir = ensure_data_dir() if download_if_missing else data_dir()
    path = raw_dir / "lalonde.csv"

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Jobs/LaLonde data not found at {path}. "
                "Run with download_if_missing=True or place the file manually."
            )
        try:
            _download(JOBS_URL, path, fallback_urls=[JOBS_FALLBACK])
        except FileNotFoundError:
            _copy_bundled("lalonde.csv", path)

    df = pd.read_csv(path)
    if "treat" in df.columns:
        T = df["treat"].values.astype(float)
    elif "treatment" in df.columns:
        T = df["treatment"].values.astype(float)
    else:
        raise ValueError(f"Jobs dataset missing treatment column. Found: {list(df.columns)}")

    if "re78" in df.columns:
        Y = df["re78"].values.astype(float)
    elif "outcome" in df.columns:
        Y = df["outcome"].values.astype(float)
    else:
        raise ValueError(f"Jobs dataset missing outcome column. Found: {list(df.columns)}")

    exclude = {
        "data_id",
        "treat",
        "treatment",
        "re78",
        "outcome",
        "re74",
        "re75",
        "nodegr",
        "nodegree",
        "u74",
        "u75",
        "hisp",
        "hispanic",
        "marr",
        "married",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(float)

    overlap = compute_overlap_diagnostics(X, T)

    return RealWorldDataset(
        X=X,
        T=T,
        Y=Y,
        tau_true=None,
        overlap_diagnostics=overlap,
        metadata={
            "name": "jobs",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        },
    )


def _load_twins_from_cevae(raw_dir: Path) -> RealWorldDataset:
    """Load and merge CEVAE TWINS CSV components."""
    twins_dir = raw_dir / "TWINS"
    twins_dir.mkdir(parents=True, exist_ok=True)
    for key, fname in TWINS_FILES.items():
        dest = twins_dir / fname
        if not dest.exists():
            _download(TWINS_BASE_URL + fname, dest)

    df_t = pd.read_csv(twins_dir / TWINS_FILES["T"], index_col=0)
    df_x = pd.read_csv(twins_dir / TWINS_FILES["X"], index_col=0)
    df_y = pd.read_csv(twins_dir / TWINS_FILES["Y"], index_col=0)

    # Lighter twin is always index 0; low birth weight is the "treatment".
    T = np.ones(len(df_y), dtype=float)
    Y = df_y["mort_0"].values.astype(float)
    y_cf = df_y["mort_1"].values.astype(float)
    tau_true = np.where(T == 1, Y - y_cf, y_cf - Y).astype(float)

    X = df_x.select_dtypes(include=[np.number]).values.astype(float)

    overlap = compute_overlap_diagnostics(X, T)
    return RealWorldDataset(
        X=X,
        T=T,
        Y=Y,
        tau_true=tau_true,
        overlap_diagnostics=overlap,
        metadata={
            "name": "twins",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "source": "cevae_twins",
        },
        ground_truth={"y_cfactual": y_cf, "ate": float(np.mean(tau_true))},
    )


def load_twins(download_if_missing: bool = True) -> RealWorldDataset:
    """
    Load Twins mortality dataset (semi-synthetic, large-n).

    Oracle CATE from counterfactual outcomes when available in source file.
    """
    raw_dir = ensure_data_dir() if download_if_missing else data_dir()
    twins_dir = raw_dir / "TWINS"
    required = [twins_dir / TWINS_FILES[k] for k in TWINS_FILES]

    if not all(p.exists() for p in required):
        if not download_if_missing:
            raise FileNotFoundError(
                f"Twins data not found under {twins_dir}. "
                "Run with download_if_missing=True or place CEVAE TWINS files manually."
            )
        try:
            return _load_twins_from_cevae(raw_dir)
        except FileNotFoundError:
            bundled = bundled_dir() / "Twin_data.csv"
            if bundled.exists():
                return _load_twins_csv(bundled)
            raise

    return _load_twins_from_cevae(raw_dir)


def _load_twins_csv(path: Path) -> RealWorldDataset:
    """Load pre-merged Twins CSV (bundled fallback format)."""
    df = pd.read_csv(path)

    if "treatment" in df.columns:
        T = df["treatment"].values.astype(float)
    elif "treat" in df.columns:
        T = df["treat"].values.astype(float)
    else:
        T = df.iloc[:, 0].values.astype(float)

    if "y_factual" in df.columns:
        Y = df["y_factual"].values.astype(float)
    elif "outcome" in df.columns:
        Y = df["outcome"].values.astype(float)
    else:
        Y = df.iloc[:, 1].values.astype(float)

    if "y_cfactual" in df.columns:
        y_cf = df["y_cfactual"].values.astype(float)
        tau_true = np.where(T == 1, Y - y_cf, y_cf - Y).astype(float)
    elif "y0" in df.columns and "y1" in df.columns:
        tau_true = (df["y1"] - df["y0"]).values.astype(float)
    else:
        feature_start = 3 if df.shape[1] > 3 else 2
        X = df.iloc[:, feature_start:].values.astype(float)
        overlap = compute_overlap_diagnostics(X, T)
        return RealWorldDataset(
            X=X,
            T=T,
            Y=Y,
            tau_true=None,
            overlap_diagnostics=overlap,
            metadata={"name": "twins", "n_samples": X.shape[0], "n_features": X.shape[1]},
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in {"treatment", "treat", "y_factual", "y_cfactual", "outcome", "y0", "y1"}
    ]
    X = df[feature_cols].values.astype(float)
    overlap = compute_overlap_diagnostics(X, T)

    return RealWorldDataset(
        X=X,
        T=T,
        Y=Y,
        tau_true=tau_true,
        overlap_diagnostics=overlap,
        metadata={
            "name": "twins",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        },
        ground_truth={"ate": float(np.mean(tau_true)) if tau_true is not None else None},
    )


def load_real_dataset(
    name: str,
    download_if_missing: bool = True,
    **kwargs: Any,
) -> RealWorldDataset:
    """Dispatch loader by dataset name: ihdp, jobs, twins."""
    loaders = {
        "ihdp": load_ihdp,
        "jobs": load_jobs,
        "twins": load_twins,
    }
    if name.lower() not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")
    return loaders[name.lower()](download_if_missing=download_if_missing, **kwargs)
