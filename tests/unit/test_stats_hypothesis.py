"""Тести гіпотез: Велч, Манн–Вітні, хі-квадрат, поправка Холма."""

import numpy as np
import pandas as pd
import pytest

from risk.stats.hypothesis import (
    COLUMNS,
    chi2_test,
    compare_continuous,
    holm,
    is_normal,
    rank_auc,
    run_all_tests,
)


def test_is_normal() -> None:
    rng = np.random.default_rng(1)
    assert is_normal(rng.normal(size=500))
    assert not is_normal(rng.exponential(size=500))
    assert not is_normal(rng.normal(size=19))  # мала вибірка


def test_welch_for_normal() -> None:
    rng = np.random.default_rng(2)
    r = compare_continuous(rng.normal(0, 1, 300), rng.normal(0.5, 1, 300), "x")
    assert r.test == "welch" and r.p < 0.05 and (r.n0, r.n1) == (300, 300)


def test_mann_whitney_for_skewed() -> None:
    rng = np.random.default_rng(3)
    r = compare_continuous(rng.exponential(1, 300), rng.exponential(3, 300), "x")
    assert r.test == "mann_whitney" and r.p < 0.05


def test_chi2_vs_manual() -> None:
    a = np.array([1] * 30 + [1] * 10 + [0] * 20 + [0] * 40)
    b = np.array([1] * 30 + [0] * 10 + [1] * 20 + [0] * 40)
    r = chi2_test(a, b, "f")
    # таблиця [[40,20],[10,30]]: chi2 = N(ad-bc)^2 / (r1 r2 c1 c2)
    manual = 100 * (40 * 30 - 20 * 10) ** 2 / (60 * 40 * 50 * 50)
    assert r.test == "chi2"
    assert r.statistic == pytest.approx(manual)
    assert (r.n0, r.n1) == (50, 50)


def test_holm_manual() -> None:
    # відсортовано: 0.01*3=0.03; 0.03*2=0.06; 0.04*1=0.04 -> max(0.06)=0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_run_all_tests() -> None:
    rng = np.random.default_rng(4)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "t2d": y,
            "age": rng.normal(50, 10, n) + 8 * y,
            "bmi": rng.exponential(5, n) + 25,
            "male": rng.integers(0, 2, n),
        }
    )
    out = run_all_tests(df, ["age", "bmi"], ["male"])
    assert len(out) == 3
    assert (out["p_holm"] >= out["p"] - 1e-12).all()
    assert out["reject_h0"].dtype == bool
    assert out.set_index("variable").loc["age", "reject_h0"]
    texts = out.set_index("variable")
    assert texts.loc["age", "test"] == "welch"
    assert (
        texts.loc["age", "h0_text"] == "H0: розподіл показника «Вік» однаковий у групах з ЦД2 і без"
    )
    assert texts.loc["bmi", "test"] == "mann_whitney"
    assert texts.loc["bmi", "h0_text"].startswith("H0: розподіл показника «Індекс маси тіла")
    assert (
        texts.loc["bmi", "h1_text"].startswith("H1: розподіл показника")
        and "відрізняється" in texts.loc["bmi", "h1_text"]
    )
    assert texts.loc["male", "h0_text"].startswith("H0: частка «Стать (чоловіча)» однакова")
    assert list(out.columns) == COLUMNS
    assert 0.5 < texts.loc["age", "auc"] < 1  # у групі з ЦД2 вік вищий
    assert texts.loc["age", "r_rb"] == pytest.approx(2 * texts.loc["age", "auc"] - 1)
    assert np.isnan(texts.loc["male", "auc"]) and np.isnan(texts.loc["male", "r_rb"])
    # частки бінарної ознаки по групах; для неперервних — NaN
    for g in (0, 1):
        expected = df.loc[df["t2d"] == g, "male"].mean() * 100
        assert texts.loc["male", f"pct{g}"] == pytest.approx(expected)
    assert texts[["pct0", "pct1"]].loc[["age", "bmi"]].isna().all().all()


def test_rank_auc_manual() -> None:
    # пари (x1, x0): 3>1, 3>2, 2>1, 2=2 → (3 + ½) / 4
    assert rank_auc(np.array([1.0, 2.0]), np.array([2.0, 3.0])) == pytest.approx(0.875)
    r = compare_continuous([1.0, 2.0], [2.0, 3.0])
    assert r.auc == pytest.approx(0.875)
    assert np.isnan(chi2_test([0, 1], [0, 1]).auc)


def test_run_all_tests_empty_returns_columns() -> None:
    out = run_all_tests(pd.DataFrame({"t2d": [0, 1]}), [], [])
    assert out.empty and list(out.columns) == COLUMNS


def test_holm_ignores_nan() -> None:
    out = holm([0.01, float("nan"), 0.04])
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.04)
    assert np.isnan(out[1])
    assert all(np.isnan(x) for x in holm([float("nan"), float("nan")]))
