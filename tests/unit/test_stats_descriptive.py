"""Описова статистика за групами."""

import numpy as np
import pandas as pd
import pytest

from risk.stats.descriptive import STAT_COLS, describe_by_group


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"t2d": rng.integers(0, 2, 200), "x": rng.normal(10, 3, 200)})
    d.loc[:9, "x"] = np.nan
    return d


def test_describe_matches_pandas(df: pd.DataFrame) -> None:
    out = describe_by_group(df, ["x"])
    assert list(out.columns) == STAT_COLS
    assert out.index.names == ["variable", "group"]
    for g, sub in df.groupby("t2d"):
        x = sub["x"].dropna()
        row = out.loc[("x", g)]
        assert row["n"] == len(x)
        assert row["mean"] == pytest.approx(x.mean())
        assert row["sd"] == pytest.approx(x.std())
        assert row["median"] == pytest.approx(x.median())
        assert row["q1"] == pytest.approx(x.quantile(0.25))
        assert row["q3"] == pytest.approx(x.quantile(0.75))
        assert row["iqr"] == pytest.approx(row["q3"] - row["q1"])
        assert (row["min"], row["max"]) == (x.min(), x.max())
