"""Матриці кореляцій Пірсона і Спірмена."""

import numpy as np
import pandas as pd
import pytest

from risk.stats.correlation import corr_matrix


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    x = rng.normal(size=300)
    return pd.DataFrame(
        {"x": x, "y": x + rng.normal(size=300), "z": np.exp(x), "t2d": (x > 0).astype(int)}
    )


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_matrix_symmetric_unit_diag(df: pd.DataFrame, method: str) -> None:
    m = corr_matrix(df, ["x", "y", "z"], method)
    assert m.shape == (3, 3)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1)


def test_spearman_monotone_invariant(df: pd.DataFrame) -> None:
    m = corr_matrix(df, ["x", "y"], "spearman")
    m2 = corr_matrix(df.assign(x=np.exp(df.x)), ["x", "y"], "spearman")
    assert np.allclose(m, m2)
    assert corr_matrix(df, ["x", "z"], "spearman").loc["x", "z"] == pytest.approx(1)
