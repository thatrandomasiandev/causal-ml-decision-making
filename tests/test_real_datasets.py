"""Tests for real-world dataset loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_ml.data.real_datasets import (
    RealWorldDataset,
    compute_overlap_diagnostics,
    load_ihdp,
    load_jobs,
    load_twins,
)


@pytest.fixture
def fixture_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect data_dir to a temp directory with synthetic fixtures."""
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr("causal_ml.data.real_datasets.data_dir", lambda: raw)
    monkeypatch.setattr("causal_ml.data.real_datasets.ensure_data_dir", lambda: raw)
    _write_ihdp_fixture(raw / "ihdp_npci_1-100.train.npz")
    _write_jobs_fixture(raw / "lalonde.csv")
    _write_twins_fixture(raw)
    return raw


def _write_ihdp_fixture(path: Path) -> None:
    n, p, reps = 50, 25, 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((reps, n, p))
    t = rng.binomial(1, 0.5, (reps, n)).astype(float)
    mu0 = rng.standard_normal((reps, n))
    mu1 = mu0 + 0.5
    yf = t * mu1 + (1 - t) * mu0
    np.savez(path, x=x, t=t, yf=yf, mu0=mu0, mu1=mu1)


def _write_jobs_fixture(path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "treat": rng.binomial(1, 0.4, n),
            "re78": rng.binomial(1, 0.5, n),
            "age": rng.integers(20, 55, n),
            "educ": rng.integers(8, 16, n),
            "black": rng.binomial(1, 0.3, n),
        }
    )
    df.to_csv(path, index=False)


def _write_twins_fixture(raw: Path) -> None:
    """Write minimal CEVAE-format TWINS files for offline tests."""
    twins_dir = raw / "TWINS"
    twins_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    n, p = 200, 10
    pd.DataFrame({"dbirwt_0": rng.integers(1000, 3000, n), "dbirwt_1": rng.integers(1000, 3000, n)}).to_csv(
        twins_dir / "twin_pairs_T_3years_samesex.csv"
    )
    X = rng.standard_normal((n, p))
    pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]).to_csv(
        twins_dir / "twin_pairs_X_3years_samesex.csv"
    )
    y0 = rng.binomial(1, 0.3, n).astype(float)
    y1 = rng.binomial(1, 0.4, n).astype(float)
    pd.DataFrame({"mort_0": y0, "mort_1": y1}).to_csv(twins_dir / "twin_pairs_Y_3years_samesex.csv")


def test_ihdp_shapes(fixture_dir: Path) -> None:
    data = load_ihdp(download_if_missing=False, realization=1)
    assert isinstance(data, RealWorldDataset)
    assert data.X.shape == (50, 25)
    assert data.T.shape == (50,)
    assert data.Y.shape == (50,)
    assert data.tau_true is not None
    assert data.X.dtype == float
    assert not np.isnan(data.X).any()


def test_jobs_shapes(fixture_dir: Path) -> None:
    data = load_jobs(download_if_missing=False)
    assert data.X.ndim == 2
    assert data.T.dtype == float
    assert data.tau_true is None
    assert not np.isnan(data.X).any()


def test_twins_tau_true(fixture_dir: Path) -> None:
    data = load_twins(download_if_missing=False)
    assert data.tau_true is not None
    assert len(data.tau_true) == data.n_samples


def test_overlap_diagnostics() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 5))
    T = rng.binomial(1, 0.5, 200)
    diag = compute_overlap_diagnostics(X, T)
    assert "propensity" in diag
    assert "ess_ratio" in diag
    assert "low_overlap" in diag
    assert 0 < diag["ess_ratio"] <= 1.0


def test_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("causal_ml.data.real_datasets.data_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="IHDP data not found"):
        load_ihdp(download_if_missing=False)
