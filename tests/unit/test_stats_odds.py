"""Відношення шансів: 2×2, одно- і багатофакторна логістична регресія."""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from risk.config import Z95
from risk.errors import DataValidationError
from risk.stats.odds import (
    ACTIVITY_LABEL,
    WAIST_LABEL,
    cholesterol_by_medication,
    clinical_or_table,
    multivariable_logit,
    or_for_threshold,
    or_from_2x2,
    per_clinical_step,
    rank_factors,
    share_by_outcome,
    univariate_or,
)


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 3000
    waist = rng.normal(95, 12, n)
    bmi = 0.3 * waist + rng.normal(0, 2, n)
    male = rng.integers(0, 2, n)
    z = -14 + 0.1 * waist + 0.1 * bmi + 0.5 * male
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"t2d": y, "waist": waist, "bmi": bmi, "male": male})


def test_or_2x2_manual() -> None:
    r = or_from_2x2(20, 10, 5, 15)
    se = math.sqrt(1 / 20 + 1 / 10 + 1 / 5 + 1 / 15)
    assert r.odds_ratio == pytest.approx(6.0)
    assert Z95 == pytest.approx(1.959964, abs=1e-6)
    assert r.ci_low == pytest.approx(math.exp(math.log(6) - Z95 * se))
    assert r.ci_high == pytest.approx(math.exp(math.log(6) + Z95 * se))
    assert r.ci_low == pytest.approx(1.693, abs=1e-3)
    assert r.ci_high == pytest.approx(21.26, abs=1e-2)
    assert r.p == pytest.approx(math.erfc(math.log(6) / se / math.sqrt(2)))  # 2·(1−Φ(z))
    assert r.n == 50


def test_or_2x2_haldane() -> None:
    r = or_from_2x2(10, 0, 5, 15)
    assert r.odds_ratio == pytest.approx(10.5 * 15.5 / (0.5 * 5.5))
    assert r.n == 30
    assert math.isfinite(r.ci_high)


def test_univariate_binary(df: pd.DataFrame) -> None:
    r = univariate_or(df, "male")
    f1, f0 = df[df.male == 1], df[df.male == 0]
    ref = or_from_2x2(
        int(f1.t2d.sum()), int((1 - f1.t2d).sum()), int(f0.t2d.sum()), int((1 - f0.t2d).sum())
    )
    assert r.odds_ratio == pytest.approx(ref.odds_ratio)
    assert r.p == pytest.approx(ref.p)


def test_univariate_continuous_vs_sklearn(df: pd.DataFrame) -> None:
    r = univariate_or(df, "waist")
    sk = LogisticRegression(C=np.inf, solver="newton-cholesky").fit(df[["waist"]], df.t2d)
    # OR на крок OR_SCALE["waist"] = 10 см
    assert r.scale == 10 and "на 10 см" in r.label
    assert r.odds_ratio == pytest.approx(math.exp(10 * sk.coef_[0][0]), abs=1e-3)
    assert r.ci_low < r.odds_ratio < r.ci_high
    assert r.n == len(df)


def test_or_for_threshold(df: pd.DataFrame) -> None:
    r = or_for_threshold(df, "waist", 100)
    hi = df.waist >= 100
    ref = or_from_2x2(
        int((hi & (df.t2d == 1)).sum()),
        int((hi & (df.t2d == 0)).sum()),
        int((~hi & (df.t2d == 1)).sum()),
        int((~hi & (df.t2d == 0)).sum()),
    )
    assert r.odds_ratio == pytest.approx(ref.odds_ratio)
    assert r.odds_ratio > 1


def test_multivariable(df: pd.DataFrame) -> None:
    df = df.copy()
    df.loc[:49, "bmi"] = np.nan
    t = multivariable_logit(df, ["waist", "bmi", "male"])
    assert list(t.index) == ["const", "waist", "bmi", "male"]
    assert np.allclose(t.odds_ratio, np.exp(t.coef))
    assert ((t.ci_low <= t.odds_ratio) & (t.odds_ratio <= t.ci_high)).all()
    assert t.attrs["converged"] and t.attrs["n"] == len(df) - 50
    assert t.attrs["n_pos"] == int(df.dropna().t2d.sum())
    assert math.isnan(t.loc["const", "vif"])
    assert t.loc["waist", "vif"] > 1 and t.loc["bmi", "vif"] > 1
    assert set(t.attrs) == {"n", "n_pos", "converged", "llf", "pseudo_r2"}


def test_continuous_ci_scaled(df: pd.DataFrame) -> None:
    r = univariate_or(df, "waist")
    fit = sm.Logit(df.t2d, sm.add_constant(df.waist)).fit(disp=0)
    lo, hi = fit.conf_int().loc["waist"]
    assert r.ci_low == pytest.approx(math.exp(10 * lo), rel=1e-4)
    assert r.ci_high == pytest.approx(math.exp(10 * hi), rel=1e-4)
    assert r.p == pytest.approx(fit.pvalues["waist"])
    assert univariate_or(df, "male").label == "Стать (чоловіча)"


def test_rank_factors(df: pd.DataFrame) -> None:
    df = df.assign(cycle_2017=(df.index % 2).astype(int))
    t = multivariable_logit(df, ["waist", "bmi", "male", "cycle_2017"])
    assert math.isnan(t.loc["const", "sd"])
    assert t.loc["waist", "std_effect"] == pytest.approx(
        abs(t.loc["waist", "coef"]) * df.waist.std()
    )
    r = rank_factors(t)
    assert "const" not in r.index and "cycle_2017" not in r.index
    assert list(r["rank"]) == [1, 2, 3]
    assert r["std_effect"].is_monotonic_decreasing
    assert "cycle_2017" in rank_factors(t, exclude=()).index


def test_clinical_or_table() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    d = pd.DataFrame(
        {
            "age": rng.normal(50, 14, n),
            "bmi": rng.normal(29, 5, n),
            "sbp": rng.normal(125, 18, n),
            "waist": rng.normal(98, 13, n),
            "male": rng.integers(0, 2, n),
            "pa_h_week": rng.exponential(3, n),
        }
    )
    z = -6 + 0.04 * d.age + 0.05 * d.bmi + 0.01 * d.sbp - 0.1 * d.pa_h_week
    d["t2d"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    t = clinical_or_table(d)
    assert len(t) == 5
    assert (t.odds_ratio > 0).all()
    assert ((t.ci_low <= t.odds_ratio) & (t.odds_ratio <= t.ci_high)).all()
    assert list(t.columns) == ["label", "odds_ratio", "ci_low", "ci_high", "p", "n_exposed", "n"]
    assert ACTIVITY_LABEL == "Активність < 3.5 год/тиждень (проти ≥ 3.5)"
    act = t.set_index("label").loc[ACTIVITY_LABEL]
    assert act.odds_ratio > 1 and act.n_exposed == (d.pa_h_week < 3.5).sum()
    w = t.set_index("label").loc[WAIST_LABEL]
    assert w.n_exposed == (np.where(d.male == 1, d.waist >= 102, d.waist >= 88)).sum()


def test_clinical_or_table_skips_missing_columns(df: pd.DataFrame) -> None:
    assert list(clinical_or_table(df).label) == ["ІМТ ≥ 30 кг/м² (проти < 30)", WAIST_LABEL]
    assert list(clinical_or_table(df.drop(columns="male")).label) == ["ІМТ ≥ 30 кг/м² (проти < 30)"]


def test_multivariable_too_few_complete_cases(df: pd.DataFrame) -> None:
    sparse = df.assign(bmi=np.where(df.index % 2 == 0, np.nan, df.bmi))
    sparse["waist"] = np.where(df.index % 2 == 1, np.nan, df.waist)  # повних випадків 0
    with pytest.raises(DataValidationError, match=r"Замало повних випадків.*\(N = 0\)"):
        multivariable_logit(sparse, ["waist", "bmi", "male"])
    with pytest.raises(DataValidationError, match=r"\(N = 39\)"):
        multivariable_logit(df.head(39), ["waist", "bmi", "male"])  # потрібно 10·(3 + 1)
    assert multivariable_logit(df.head(40), ["waist", "bmi", "male"]).attrs["n"] == 40


def test_per_clinical_step(df: pd.DataFrame) -> None:
    age = np.random.default_rng(1).normal(50, 10, len(df))
    t = multivariable_logit(df.assign(age=age), ["waist", "male", "age"])
    s = per_clinical_step(t)
    assert "const" not in s.index
    assert s.loc["waist", "odds_ratio"] == pytest.approx(math.exp(10 * t.loc["waist", "coef"]))
    assert s.loc["waist", "ci_low"] == pytest.approx(t.loc["waist", "ci_low"] ** 10)
    assert s.loc["male", "odds_ratio"] == t.loc["male", "odds_ratio"]
    assert list(s["label"]) == [
        "Окружність талії, на 10 см",
        "Стать (чоловіча)",
        "Вік, на 10 років",
    ]


def test_cholesterol_by_medication() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    med = rng.integers(0, 2, n)
    t2d = (rng.random(n) < np.where(med == 1, 0.4, 0.1)).astype(int)
    tc = rng.normal(200, 30, n) - 30 * med  # ліки знижують холестерин, зв'язку в групах немає
    d = pd.DataFrame({"tc": tc, "chol_med": med.astype(float), "t2d": t2d})
    d.loc[:9, "chol_med"] = np.nan
    t = cholesterol_by_medication(d)
    assert list(t.columns) == ["group", "n", "n_pos", "odds_ratio", "ci_low", "ci_high", "p"]
    assert t["n"].tolist() == [n - 10, int((d.chol_med == 0).sum()), int((d.chol_med == 1).sum())]
    assert t.loc[0, "odds_ratio"] < 1  # загалом «обернений» зв'язок
    assert t.loc[1, "ci_low"] < 1 < t.loc[1, "ci_high"]  # без ліків зв'язку немає
    ref = univariate_or(d[d.chol_med == 0], "tc")
    assert t.loc[1, "odds_ratio"] == pytest.approx(ref.odds_ratio)


def test_share_by_outcome() -> None:
    d = pd.DataFrame({"x": [1, 0, 1, 1, np.nan], "t2d": [1, 1, 0, 0, 1]})
    assert share_by_outcome(d, "x") == (50.0, 100.0)
