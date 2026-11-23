"""Зважена поширеність: SE, комбіновані ваги, design effect."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk.config import CYCLES
from risk.data.weights import design_effect, prevalence, survey_mean_se


def test_se_equals_simple_sample() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=50))
    w = pd.Series(np.ones(50))
    p, se = survey_mean_se(y, w, pd.Series(np.zeros(50)), pd.Series(np.arange(50)))
    assert p == pytest.approx(y.mean())
    assert se == pytest.approx(stats.sem(y))


def test_weighted_mean_matches_average() -> None:
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 40).astype(float))
    w = pd.Series(rng.uniform(1, 5, 40))
    strata = pd.Series(np.repeat([1, 2], 20))
    psu = pd.Series(np.tile([1, 2], 20))
    p, se = survey_mean_se(y, w, strata, psu)
    assert p == pytest.approx(np.average(y, weights=w))
    assert se > 0


def test_combined_weights_sum_to_one() -> None:
    assert sum(c["weight_k"] for c in CYCLES.values()) == pytest.approx(1.0)


def make_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "t2d": rng.integers(0, 2, 60).astype(float),
            "wt_mec": rng.uniform(1, 5, 60),
            "strata": np.repeat([1, 2, 3], 20),
            "psu": np.tile([1, 2], 30),
            "cycle": np.repeat(["2015-2016", "2017-2020"], 30),
        }
    )


def test_prevalence_total_and_by() -> None:
    df = make_df()
    df.loc[0, "t2d"] = np.nan
    total = prevalence(df)
    assert list(total.columns) == [
        "group", "n", "unweighted_pct", "weighted_pct", "ci_low", "ci_high",
    ]  # fmt: skip
    assert total["group"][0] == "Разом"
    assert total["n"][0] == 59
    assert total["ci_low"][0] < total["weighted_pct"][0] < total["ci_high"][0]
    by = prevalence(df, by="cycle")
    assert by["group"].tolist() == ["2015-2016", "2017-2020"]
    assert by["n"].sum() == 59


def test_prevalence_drops_zero_weights() -> None:
    df = make_df()
    df.loc[:4, "wt_mec"] = 0.0
    assert prevalence(df)["n"][0] == 55


def test_design_effect_simple_random_sample() -> None:
    # Рівні ваги, кожен учасник — окремий PSU однієї страти: deff = n / (n − 1)
    n = 40
    df = pd.DataFrame(
        {"t2d": np.r_[np.ones(10), np.zeros(n - 10)], "wt_mec": 1.0, "strata": 1,
         "psu": np.arange(n)}
    )  # fmt: skip
    deff, n_eff = design_effect(df)
    assert deff == pytest.approx(n / (n - 1))
    assert n_eff == pytest.approx(n - 1)


def test_design_effect_clustered_above_one() -> None:
    df = make_df()
    df["t2d"] = (df["psu"] == 1).astype(float)  # результат повністю визначається кластером
    assert design_effect(df)[0] > 1
