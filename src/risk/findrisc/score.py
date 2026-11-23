"""Модифікований FINDRISC: 7 пунктів, максимум 25 балів.

Пороги — за Lindström & Tuomilehto (2003); пункт «овочі/фрукти» виключено, бо в NHANES
немає відповідного питання. Пункт про глюкозу в анамнезі має три варіанти через ризик витоку.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from risk.config import ACTIVE_MIN_PER_WEEK, WAIST_CUTS_CM
from risk.errors import DataValidationError

MAX_SCORE_A = 20  # максимум балів варіанта (a): шість пунктів без глюкози в анамнезі
CATEGORY_BOUNDS = [(0, 6), (7, 11), (12, 14), (15, 20), (21, 25)]
CATEGORY_NAMES = ["низький", "трохи підвищений", "помірний", "високий", "дуже високий"]
ITEM_COLUMNS = [
    "item_age",
    "item_bmi",
    "item_waist",
    "item_activity",
    "item_bp_med",
    "item_glucose",
    "item_family",
]


# ---------- скалярні пункти ----------
def age_points(age: float) -> int:
    """<45→0; 45–54→2; 55–64→3; ≥65→4."""
    if age < 45:
        return 0
    if age < 55:
        return 2
    return 3 if age < 65 else 4


def bmi_points(bmi: float) -> int:
    """<25→0; 25–30 включно→1; >30→3."""
    if bmi < 25:
        return 0
    return 1 if bmi <= 30 else 3


def waist_points(waist: float, male: bool) -> int:
    """Чол.: <94→0; 94–102→3; >102→4. Жін.: <80→0; 80–88→3; >88→4 (верхня межа включно)."""
    low, high = WAIST_CUTS_CM[male]
    if waist < low:
        return 0
    return 3 if waist <= high else 4


def activity_points(active: bool) -> int:
    """Фізична активність: активний (≥ ACTIVE_MIN_PER_WEEK хв/тиждень) → 0; інакше 2."""
    return 0 if active else 2


def bp_med_points(takes: bool) -> int:
    """Ліки від артеріального тиску: приймає → 2; ні → 0."""
    return 2 if takes else 0


def high_glucose_points(had: bool) -> int:
    """Підвищена глюкоза в анамнезі: була → 5; ні → 0."""
    return 5 if had else 0


def family_points(has: bool) -> int:
    """Діабет у батьків, братів чи сестер → 5; ні → 0 (родичі 2-го ступеня не враховані)."""
    return 5 if has else 0


# ---------- векторні обгортки ----------
def _map(s: pd.Series, f: Callable[[float], int]) -> pd.Series:
    """Застосувати скалярну функцію, зберігаючи NaN."""
    return s.map(f, na_action="ignore").astype(float)


def _waist_items(df: pd.DataFrame) -> pd.Series:
    vals = [
        waist_points(w, bool(m)) if pd.notna(w) and pd.notna(m) else float("nan")
        for w, m in zip(df["waist"], df["male"], strict=True)
    ]
    return pd.Series(vals, index=df.index, dtype=float)


def activity_min_week(df: pd.DataFrame) -> pd.Series:
    """Активність, хв/тиждень: pa_min_week, а якщо його немає — pa_h_week × 60 (шаблон)."""
    if "pa_min_week" in df:
        return df["pa_min_week"]
    if "pa_h_week" in df:
        return df["pa_h_week"] * 60
    raise DataValidationError("Для FINDRISC потрібна колонка pa_h_week (або pa_min_week)")


def score_items(df: pd.DataFrame) -> pd.DataFrame:
    """Бали по 7 пунктах; NaN на вході → NaN у пункті."""
    minutes = activity_min_week(df)
    active = (minutes >= ACTIVE_MIN_PER_WEEK).where(minutes.notna())
    return pd.DataFrame(
        {
            "item_age": _map(df["age"], age_points),
            "item_bmi": _map(df["bmi"], bmi_points),
            "item_waist": _waist_items(df),
            "item_activity": _map(active, lambda a: activity_points(bool(a))),
            "item_bp_med": _map(df["bp_med"], lambda v: bp_med_points(bool(v))),
            "item_glucose": _map(df["high_glu_hist"], lambda v: high_glucose_points(bool(v))),
            "item_family": _map(df["fam_dm"], lambda v: family_points(bool(v))),
        },
        index=df.index,
    )


def total_score(items: pd.DataFrame, include_glucose: bool) -> pd.Series:
    """Сума пунктів без пропусків: макс. 20 без пункту про глюкозу, 25 з ним."""
    cols = ITEM_COLUMNS if include_glucose else [c for c in ITEM_COLUMNS if c != "item_glucose"]
    return items[cols].sum(axis=1, skipna=False)


def category(score: float) -> str:
    """Категорія ризику за сумою балів."""
    for (lo, hi), name in zip(CATEGORY_BOUNDS, CATEGORY_NAMES, strict=True):
        if lo <= score <= hi:
            return name
    raise DataValidationError(f"Бал FINDRISC поза діапазоном 0–25: {score}")


def categories(score: pd.Series) -> pd.Series:
    """Категорія ризику для кожного балу; NaN лишається NaN."""
    return score.map(category, na_action="ignore")


def reachable_categories(max_score: int | None = None) -> list[str]:
    """Категорії, досяжні при максимумі балів варіанта (20 без пункту про глюкозу)."""
    return [
        n
        for n, (lo, _) in zip(CATEGORY_NAMES, CATEGORY_BOUNDS, strict=True)
        if max_score is None or lo <= max_score
    ]


def category_table(score: pd.Series, y: pd.Series, max_score: int | None = None) -> pd.DataFrame:
    """Рядок на категорію (у фіксованому порядку): N, з ЦД2, % з ЦД2.

    Недосяжні при max_score категорії не виводяться; порожня категорія має pct_t2d = NaN.
    """
    cat = pd.Categorical(categories(score), categories=reachable_categories(max_score))
    g = y.groupby(cat, observed=False)
    out = pd.DataFrame({"n": g.size(), "n_t2d": g.sum()})
    out["pct_t2d"] = (100 * out["n_t2d"] / out["n"]).where(out["n"] > 0).astype(float)
    return out.rename_axis("category").reset_index().astype({"category": str})


# ---------- варіанти пункту про глюкозу в анамнезі ----------
VARIANT_TITLES = {
    "a": "(a) 6 пунктів без «глюкози в анамнезі»; тестова вибірка з повною анкетою",
    "b": "(b) скринінг: з пунктом про глюкозу, лише недіагностовані",
    "c": "(c) з витоком — лише демонстрація",
}


@dataclass(frozen=True, slots=True)
class FindriscVariant:
    """Бали і ціль варіанта на рядках без пропусків; варіант з витоком непридатний."""

    key: str
    title: str
    score: pd.Series
    y: pd.Series
    is_valid_for_comparison: bool


def _make(key: str, title: str, score: pd.Series, y: pd.Series, valid: bool) -> FindriscVariant:
    ok = score.notna() & y.notna()
    return FindriscVariant(key, title, score[ok], y[ok], valid)


def variants(cohort: pd.DataFrame) -> dict[str, FindriscVariant]:
    """Три варіанти FINDRISC: a (без п.7), b (скринінг), c (з витоком)."""
    items = score_items(cohort)
    diag = cohort["diq010"] == 1
    leaky = items.assign(item_glucose=items["item_glucose"].where(~diag, 5.0))
    undiag = ~diag
    return {
        "a": _make(
            "a", VARIANT_TITLES["a"],
            total_score(items, False), cohort["t2d"], True,
        ),
        "b": _make(
            "b", VARIANT_TITLES["b"],
            total_score(items, True)[undiag], cohort["t2d_undiag"][undiag], True,
        ),
        "c": _make(
            "c", VARIANT_TITLES["c"],
            total_score(leaky, True), cohort["t2d"], False,
        ),
    }  # fmt: skip


# ---------- таблиця «оригінал vs модифікований» ----------
ORIGINAL_VS_MODIFIED: list[dict] = [
    {"item": "Вік", "original": "0/2/3/4", "modified": "0/2/3/4 (<45; 45–54; 55–64; ≥65)",
     "change": "Без змін. У NHANES вік обрізано на 80 роках."},
    {"item": "ІМТ", "original": "0/1/3", "modified": "0/1/3 (<25; 25–30; >30)",
     "change": "Без змін; рівно 30 → 1 бал."},
    {"item": "Окружність талії", "original": "0/3/4",
     "modified": "0/3/4 (чол. <94; 94–102; >102; жін. <80; 80–88; >88)",
     "change": "Без змін; верхня межа включно."},
    {"item": "Фізична активність", "original": "0/2", "modified": "0/2",
     "change": f"Критерій «≥ 30 хв/день» замінено на ≥ {ACTIVE_MIN_PER_WEEK} хв/тиждень "
               "помірної або інтенсивної активності за анкетою (GPAQ, PAQ605–PAD675)."},
    {"item": "Овочі/фрукти щодня", "original": "0/1", "modified": "виключено",
     "change": "У NHANES немає такого питання; показник з опитування про їжу за добу "
               "дав обернений зв'язок, тому пункт виключено."},
    {"item": "Ліки від тиску", "original": "0/2", "modified": "0/2",
     "change": "В оригіналі — «колись регулярно приймав»; тут — «приймає зараз» "
               "(BPQ050A)."},
    {"item": "Висока глюкоза в анамнезі", "original": "0/5", "modified": "0/5",
     "change": "В оригіналі враховано й діабет вагітних; тут — «лікар казав про предіабет» "
               "(DIQ160). Три варіанти (a/b/c) через ризик витоку для діагностованих."},
    {"item": "Спадковість", "original": "0/3/5", "modified": "0/5",
     "change": "Питання про діабет у батьків, братів і сестер (MCQ300C) відповідає 5 балам "
               "оригіналу; родичі 2-го ступеня (3 бали) не враховані."},
    {"item": "Максимум балів", "original": "26", "modified": "25 (20 без пункту про глюкозу)",
     "change": "Виключено пункт «овочі/фрукти» (1 бал)."},
]  # fmt: skip
