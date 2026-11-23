"""Фільтри когорти, потік відбору і примітки до нього."""

import pandas as pd

from risk.data import cohort as c


def synthetic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 25, 30, 40, 50, 60],
            "pregnant": [False, True, False, False, False, False],
            "hba1c": [5.0, 5.0, None, 5.0, 5.0, 5.0],
            "t2d": [0.0, 0.0, 1.0, None, 0.0, 1.0],
        }
    )


def test_each_filter() -> None:
    df = synthetic()
    assert len(c.filter_adults(df)) == 5
    assert len(c.filter_not_pregnant(df)) == 5
    assert len(c.filter_hba1c_measured(df)) == 5
    assert len(c.filter_target_defined(df)) == 5


def test_build_cohort_flow() -> None:
    cohort, steps = c.build_cohort(synthetic())
    assert [s.n for s in steps] == [6, 5, 4, 3, 2]
    assert [s.excluded for s in steps] == [0, 1, 1, 1, 1]
    assert [s.label for s in steps] == list(c.LABELS)
    assert cohort["t2d"].notna().all()
    assert cohort["age"].tolist() == [50, 60]


def test_cohort_notes() -> None:
    harmonized = pd.DataFrame(
        {
            "age": [10, 30, 40, 50, 60, 70, 35],
            "pregnant": [False] * 6 + [True],
            "hba1c": [None, None, None, None, 5.0, 7.0, None],
            "examined": [False, False, True, True, True, True, False],
            "diq010": [1, 1, 2, 1, 2, 1, 1],
            "t2d": [None, None, None, None, 0.0, 1.0, None],
            "t1d_proxy": [None, 1.0, None, 0.0, None, 1.0, None],
        }
    )
    cohort, _ = c.build_cohort(harmonized)
    assert c.cohort_notes(harmonized, cohort) == {
        "excl_no_mec": 1,
        "excl_examined_no_hba1c": 2,
        "excl_hba1c_diagnosed": 2,
        "diagnosed_n": 1,
        "t1d_proxy_n": 1,
        "t1d_proxy_pct": 100.0,
    }


def test_cohort_notes_without_diagnosed() -> None:
    df = pd.DataFrame(
        {"age": [30], "pregnant": [False], "hba1c": [5.0], "examined": [True], "diq010": [2],
         "t2d": [0.0], "t1d_proxy": [None]}
    )  # fmt: skip
    assert c.cohort_notes(df, df)["t1d_proxy_pct"] == 0.0
