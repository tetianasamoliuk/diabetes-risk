"""Тести модифікованого FINDRISC: межі порогів, варіанти a/b/c, запобіжник."""

import numpy as np
import pandas as pd
import pytest

from risk.config import ACTIVE_MIN_PER_WEEK
from risk.errors import DataValidationError
from risk.findrisc.score import (
    CATEGORY_NAMES,
    ORIGINAL_VS_MODIFIED,
    activity_points,
    age_points,
    bmi_points,
    bp_med_points,
    categories,
    category,
    category_table,
    family_points,
    high_glucose_points,
    score_items,
    total_score,
    variants,
    waist_points,
)

ITEM_COLS = [
    "item_age",
    "item_bmi",
    "item_waist",
    "item_activity",
    "item_bp_med",
    "item_glucose",
    "item_family",
]


@pytest.mark.parametrize(
    ("age", "pts"),
    [(20, 0), (44.9, 0), (45, 2), (54.9, 2), (55, 3), (64.9, 3), (65, 4), (80, 4)],
)
def test_age_points(age: float, pts: int) -> None:
    assert age_points(age) == pts


@pytest.mark.parametrize(("bmi", "pts"), [(24.9, 0), (25, 1), (30, 1), (30.1, 3)])
def test_bmi_points(bmi: float, pts: int) -> None:
    assert bmi_points(bmi) == pts


@pytest.mark.parametrize(("waist", "pts"), [(93.9, 0), (94, 3), (102, 3), (102.1, 4)])
def test_waist_men(waist: float, pts: int) -> None:
    assert waist_points(waist, True) == pts


@pytest.mark.parametrize(("waist", "pts"), [(79.9, 0), (80, 3), (88, 3), (88.1, 4)])
def test_waist_women(waist: float, pts: int) -> None:
    assert waist_points(waist, False) == pts


def test_binary_points() -> None:
    assert (activity_points(True), activity_points(False)) == (0, 2)
    assert (bp_med_points(True), bp_med_points(False)) == (2, 0)
    assert (high_glucose_points(True), high_glucose_points(False)) == (5, 0)
    assert (family_points(True), family_points(False)) == (5, 0)


def _row(**kw: float) -> dict[str, float]:
    base = dict(
        age=50, male=1, bmi=27, waist=100, pa_min_week=300, bp_med=0,
        high_glu_hist=0, fam_dm=0, diq010=2, t2d=0, t2d_undiag=0,
    )  # fmt: skip
    return {**base, **kw}


def test_score_items_values_and_nan() -> None:
    df = pd.DataFrame(
        [
            _row(),
            _row(male=0, waist=90, pa_min_week=ACTIVE_MIN_PER_WEEK - 1, bp_med=1,
                 high_glu_hist=1, fam_dm=1),
            _row(bmi=np.nan, waist=np.nan, pa_min_week=np.nan, fam_dm=np.nan),
        ]
    )  # fmt: skip
    items = score_items(df)
    assert list(items.columns) == ITEM_COLS
    assert items.iloc[0].tolist() == [2, 1, 3, 0, 0, 0, 0]
    assert items.iloc[1].tolist() == [2, 1, 4, 2, 2, 5, 5]
    r = items.iloc[2]
    assert r[["item_bmi", "item_waist", "item_activity", "item_family"]].isna().all()
    assert r[["item_age", "item_bp_med", "item_glucose"]].notna().all()
    # межа активності: рівно 210 хв/тиж — активний
    assert score_items(pd.DataFrame([_row(pa_min_week=ACTIVE_MIN_PER_WEEK)])).item_activity[0] == 0


def test_total_score_max_and_nan() -> None:
    items = pd.DataFrame([[4, 3, 4, 2, 2, 5, 5], [4, 3, 4, 2, 2, 5, np.nan]], columns=ITEM_COLS)
    t7 = total_score(items, include_glucose=True)
    t6 = total_score(items, include_glucose=False)
    assert t7[0] == 25 and t6[0] == 20
    assert np.isnan(t7[1]) and np.isnan(t6[1])


@pytest.mark.parametrize(
    ("s", "cat"),
    [(0, "низький"), (6, "низький"), (7, "трохи підвищений"), (11, "трохи підвищений"),
     (12, "помірний"), (14, "помірний"), (15, "високий"), (20, "високий"),
     (21, "дуже високий"), (25, "дуже високий")],
)  # fmt: skip
def test_category(s: int, cat: str) -> None:
    assert category(s) == cat


def test_category_out_of_range() -> None:
    with pytest.raises(DataValidationError):
        category(26)
    with pytest.raises(DataValidationError):
        category(-1)


def test_categories_vector_nan() -> None:
    out = categories(pd.Series([3.0, np.nan, 21.0]))
    assert out[0] == "низький" and np.isnan(out[1]) and out[2] == "дуже високий"


@pytest.fixture
def cohort() -> pd.DataFrame:
    # 20 рядків: 6 діагностованих (diq010 == 1), 14 інших; у 1 рядку NaN у талії
    rows = []
    for i in range(20):
        diag = i < 6
        rows.append(
            _row(
                age=30 + 2 * i, bmi=22 + i, waist=85 + i, male=i % 2,
                high_glu_hist=int(i % 3 == 0), fam_dm=int(i % 4 == 0),
                diq010=1 if diag else 2, t2d=int(diag or i == 10),
                t2d_undiag=0 if diag else int(i == 10),
            )
        )  # fmt: skip
    df = pd.DataFrame(rows)
    df.loc[19, "waist"] = np.nan
    return df


def test_variants_sizes_and_flags(cohort: pd.DataFrame) -> None:
    v = variants(cohort)
    assert set(v) == {"a", "b", "c"}
    assert (len(v["a"].score), len(v["b"].score), len(v["c"].score)) == (19, 13, 19)
    for var in v.values():
        assert var.score.index.equals(var.y.index)
        assert var.score.notna().all()
    assert v["a"].is_valid_for_comparison and v["b"].is_valid_for_comparison
    assert not v["c"].is_valid_for_comparison
    assert v["a"].y.equals(cohort.t2d[v["a"].y.index])
    assert v["b"].y.equals(cohort.t2d_undiag[v["b"].y.index])
    assert v["c"].title.startswith("(c)")


def test_variant_b_has_no_diagnosed(cohort: pd.DataFrame) -> None:
    idx = variants(cohort)["b"].score.index
    assert (cohort.loc[idx, "diq010"] != 1).all()


def test_variant_c_diagnosed_get_5(cohort: pd.DataFrame) -> None:
    c = variants(cohort)["c"]
    items = score_items(cohort)
    diag = cohort.index[(cohort.diq010 == 1)].intersection(c.score.index)
    expected = total_score(
        items.assign(item_glucose=items.item_glucose.where(cohort.diq010 != 1, 5)), True
    )
    assert (c.score[diag] == expected[diag]).all()
    # різниця з "b"-формулою саме на п.7 у діагностованих
    plain = total_score(items, True)[diag]
    assert (c.score[diag] >= plain).all()


def test_original_vs_modified_table() -> None:
    assert ORIGINAL_VS_MODIFIED
    for r in ORIGINAL_VS_MODIFIED:
        assert {"item", "original", "modified", "change"} <= r.keys()
    assert any("MCQ300C" in r["change"] for r in ORIGINAL_VS_MODIFIED)
    assert not any("(D-" in r["change"] for r in ORIGINAL_VS_MODIFIED)  # текст для користувача


def test_category_table_order_and_counts() -> None:
    score = pd.Series([1.0, 2.0, 8.0, 22.0, np.nan])
    y = pd.Series([0, 1, 1, 1, 1])
    t = category_table(score, y)
    assert t["category"].tolist() == CATEGORY_NAMES
    assert t["n"].tolist() == [2, 1, 0, 0, 1]
    assert t["n_t2d"].tolist() == [1, 1, 0, 0, 1]
    assert t["pct_t2d"].iloc[0] == 50.0 and np.isnan(t["pct_t2d"].iloc[2])


def test_category_table_max_score_drops_unreachable() -> None:
    score = pd.Series([1.0, 2.0, 8.0, 18.0])
    y = pd.Series([0, 1, 1, 1])
    t = category_table(score, y, max_score=20)
    assert t["category"].tolist() == CATEGORY_NAMES[:4]
    assert "дуже високий" not in t["category"].tolist()
    assert t["pct_t2d"].dtype == float and np.isnan(t["pct_t2d"].iloc[2])
    assert len(category_table(score, y)) == 5


def test_variant_a_title_names_test_sample(cohort: pd.DataFrame) -> None:
    title = "(a) 6 пунктів без «глюкози в анамнезі»; тестова вибірка з повною анкетою"
    assert variants(cohort)["a"].title == title


def test_activity_from_pa_h_week_when_no_pa_min_week() -> None:
    """Файл за шаблоном має лише pa_h_week: 3.5 год = 210 хв → активний (0 балів)."""
    df = pd.DataFrame(
        {
            "age": [30.0, 30.0], "male": [1.0, 1.0], "bmi": [22.0, 22.0], "waist": [80.0, 80.0],
            "pa_h_week": [3.5, 3.4], "bp_med": [0.0, 0.0],
            "high_glu_hist": [np.nan, np.nan], "fam_dm": [0.0, 0.0],
        }
    )  # fmt: skip
    assert score_items(df)["item_activity"].tolist() == [0.0, 2.0]
    with pytest.raises(DataValidationError):
        score_items(df.drop(columns="pa_h_week"))
