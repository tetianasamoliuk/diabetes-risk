"""Чиста логіка форми «Оцінка ризику»: ІМТ, валідація, рядки ознак, сценарії «що якщо».

Сценарії — демонстрація поведінки моделі, а не медична рекомендація (див. ``config.DISCLAIMER``).
"""

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass
from typing import TypedDict, Unpack

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from risk import interpret
from risk.config import ACTIVE_MIN_PER_WEEK, FEATURES
from risk.data.user_data import RANGES, _human
from risk.errors import DataValidationError
from risk.findrisc.score import category, score_items, total_score

# Діапазони полів форми, яких немає в RANGES
PERSON_RANGES: dict[str, tuple[float, float]] = {
    "height_cm": (100, 250),
    "weight_kg": (30, 300),
    "pa_min_week": (0, 10_080),
}

# (поле, назва в повідомленні, ключ діапазону, одиниця, необов'язкове)
_CHECKS: list[tuple[str, str, str, str, bool]] = [
    ("age", "Вік", "age", "років", False),
    ("height_cm", "Зріст", "height_cm", "см", False),
    ("weight_kg", "Вага", "weight_kg", "кг", False),
    ("waist_cm", "Окружність талії", "waist", "см", False),
    ("sbp", "Систолічний тиск", "sbp", "мм рт. ст.", True),
    ("dbp", "Діастолічний тиск", "dbp", "мм рт. ст.", True),
    ("tc", "Загальний холестерин", "tc", "мг/дл", True),
    ("hdl", "ЛПВЩ-холестерин", "hdl", "мг/дл", True),
    ("pa_min_week", "Фізична активність", "pa_min_week", "хв/тиждень", False),
]


CHANGE_COL = "Зміна (основна модель), п.п."
BMI_SLIDER = (18.5, 40.0)  # межі слайдера цільового ІМТ
BMI_STEP = 0.1


def field_range(key: str) -> tuple[float, float]:
    """Допустимі межі поля форми: PERSON_RANGES, а для решти — RANGES шаблону файлу."""
    return PERSON_RANGES[key] if key in PERSON_RANGES else RANGES[key]


@dataclass(frozen=True, slots=True)
class PersonInput:
    """Введення форми.

    Необов'язкові sbp, dbp, tc, hdl («не знаю» = None): ``to_feature_row`` підставляє
    медіани навчальної вибірки (``train_medians``) однаково для всіх моделей
    і SHAP. Для FINDRISC вони не потрібні. None в обов'язковому полі — порожнє поле
    форми: ``validate`` про нього повідомляє, розрахунки такого введення не приймають.
    """

    age: float | None
    male: bool
    height_cm: float | None
    weight_kg: float | None
    waist_cm: float | None
    sbp: float | None
    dbp: float | None
    tc: float | None
    hdl: float | None
    pa_min_week: float | None
    fam_dm: bool
    bp_med: bool
    high_glu_hist: bool


class PersonChanges(TypedDict, total=False):
    """Поля PersonInput, які сценарій може замінити напряму."""

    age: float
    male: bool
    height_cm: float
    weight_kg: float
    waist_cm: float
    sbp: float | None
    dbp: float | None
    tc: float | None
    hdl: float | None
    pa_min_week: float
    fam_dm: bool
    bp_med: bool
    high_glu_hist: bool


def bmi_from(height_cm: float, weight_kg: float) -> float:
    """ІМТ, кг/м² (без округлення)."""
    if height_cm <= 0:
        raise DataValidationError(f"Зріст має бути додатним, отримано {height_cm:g} см")
    return weight_kg / (height_cm / 100) ** 2


def _field_error(
    person: PersonInput, check: tuple[str, str, str, str, bool], empty: Collection[str]
) -> str | None:
    """Повідомлення про одне поле: порожнє або поза межами; None — поле гаразд."""
    field, label, key, unit, optional = check
    value = getattr(person, field)
    if value is None:
        if not optional:
            return f"{label}: введіть значення"
        return f"{label}: введіть значення або позначте «не знаю»" if field in empty else None
    lo, hi = field_range(key)
    if lo <= value <= hi:
        return None
    return f"{label} має бути від {_human(lo)} до {_human(hi)} {unit}"


def validate(person: PersonInput, empty: Collection[str] = ()) -> list[str]:
    """Людські повідомлення про всі помилки; порожній список = валідно.

    ``empty`` — необов'язкові поля, які лишили порожніми без «не знаю»: їхній None
    означає пропуск, а не «не знаю».
    """
    errors = [msg for check in _CHECKS if (msg := _field_error(person, check, empty))]
    if person.sbp is not None and person.dbp is not None and person.dbp >= person.sbp:
        errors.append("Діастолічний тиск має бути меншим за систолічний")
    if person.tc is not None and person.hdl is not None and person.tc < person.hdl:
        errors.append("Загальний холестерин не може бути меншим за ЛПВЩ-холестерин")
    errors += _bmi_errors(person)
    return errors


def _bmi_errors(person: PersonInput) -> list[str]:
    # порожній чи недодатний зріст або порожня вага: ІМТ не рахується, про поля вже сказано
    if person.height_cm is None or person.weight_kg is None or person.height_cm <= 0:
        return []
    bmi = bmi_from(person.height_cm, person.weight_kg)
    lo, hi = RANGES["bmi"]
    if lo <= bmi <= hi:
        return []
    return [f"Отриманий ІМТ {bmi:.1f} поза правдоподібним діапазоном — перевірте зріст і вагу"]


def train_medians(models: dict[str, Pipeline]) -> dict[str, float]:
    """Медіани навчальної вибірки по FEATURES з fitted-імпутера (logit/RF)."""
    for m in models.values():
        if "impute" in m.named_steps:
            imp = m.named_steps["impute"]
            names = list(imp.feature_names_in_)
            if names != FEATURES:
                raise DataValidationError(f"Порядок колонок імпутера {names} ≠ FEATURES")
            return dict(zip(names, map(float, imp.statistics_), strict=True))
    raise DataValidationError("У жодному пайплайні немає кроку імпутації для медіан")


def _nan(value: float | None, fill: float | None = None) -> float:
    if value is None:
        return float("nan") if fill is None else float(fill)
    return float(value)


def to_feature_row(
    person: PersonInput, cycle_2017: float = 1.0, medians: dict[str, float] | None = None
) -> pd.DataFrame:
    """Один рядок з колонками config.FEATURES (у тому ж порядку).

    None в sbp/dbp/tc/hdl → медіана з ``medians`` (якщо передано), інакше NaN.
    """
    fill = medians or {}
    row = {
        "age": person.age,
        "male": float(person.male),
        "bmi": bmi_from(person.height_cm, person.weight_kg),
        "waist": person.waist_cm,
        "sbp": _nan(person.sbp, fill.get("sbp")),
        "dbp": _nan(person.dbp, fill.get("dbp")),
        "tc": _nan(person.tc, fill.get("tc")),
        "hdl": _nan(person.hdl, fill.get("hdl")),
        "pa_h_week": person.pa_min_week / 60,
        "fam_dm": float(person.fam_dm),
        "bp_med": float(person.bp_med),
        "cycle_2017": cycle_2017,
    }
    return pd.DataFrame([row], columns=FEATURES)


def to_findrisc_row(person: PersonInput) -> pd.DataFrame:
    """Один рядок з колонками, які очікує findrisc.score.score_items."""
    return pd.DataFrame(
        [
            {
                "age": person.age,
                "male": float(person.male),
                "bmi": bmi_from(person.height_cm, person.weight_kg),
                "waist": person.waist_cm,
                "pa_min_week": person.pa_min_week,
                "bp_med": float(person.bp_med),
                "high_glu_hist": float(person.high_glu_hist),
                "fam_dm": float(person.fam_dm),
            }
        ]
    )


def predict_all(
    models: dict[str, Pipeline],
    person: PersonInput,
    cycle_2017: float = 1.0,
    medians: dict[str, float] | None = None,
) -> dict[str, float]:
    """Ймовірність класу 1 від кожної моделі."""
    row = to_feature_row(person, cycle_2017, medians)
    return {name: float(m.predict_proba(row)[0, 1]) for name, m in models.items()}


def findrisc_result(person: PersonInput) -> tuple[int, str]:
    """Бал FINDRISC (усі 7 пунктів) і категорія."""
    items = score_items(to_findrisc_row(person))
    score = int(total_score(items, include_glucose=True).iloc[0])
    return score, category(score)


def scenario_table(
    models: dict[str, Pipeline],
    person: PersonInput,
    scenarios: list[dict],
    best: str,
    cycle_2017: float = 1.0,
    medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """«Поточний стан» + рядок на сценарій: ймовірності моделей (%) і зміна для best (п.п.).

    Сценарії, після яких ``validate`` дає помилки (вихід за діапазон), пропускаються;
    їхні назви — у ``df.attrs["skipped"]``.
    """
    base = predict_all(models, person, cycle_2017, medians)
    rows = [{"Сценарій": "Поточний стан", **base}]
    skipped: list[str] = []
    for s in scenarios:
        changed = apply_scenario(person, **s["changes"])
        if validate(changed):
            skipped.append(s["name"])
            continue
        rows.append({"Сценарій": s["name"], **predict_all(models, changed, cycle_2017, medians)})
    df = pd.DataFrame(rows)
    df[CHANGE_COL] = (df[best] - base[best]) * 100
    df[list(models)] = df[list(models)] * 100
    df.attrs["skipped"] = skipped
    return df


def apply_scenario(
    person: PersonInput,
    *,
    bmi_target: float | None = None,
    active: bool | None = None,
    waist_delta: float | None = None,
    **changes: Unpack[PersonChanges],
) -> PersonInput:
    """Копія person зі змінами; невідоме поле → TypeError (від dataclasses.replace).

    Спецпараметри: bmi_target (міняє вагу, не зріст), active (True → ≥ ACTIVE_MIN_PER_WEEK,
    False → 0), waist_delta (відносно поточної талії).
    """
    if bmi_target is not None:
        changes["weight_kg"] = bmi_target * (person.height_cm / 100) ** 2
    if active is not None:
        changes["pa_min_week"] = max(person.pa_min_week, ACTIVE_MIN_PER_WEEK) if active else 0.0
    if waist_delta is not None:
        changes["waist_cm"] = person.waist_cm + waist_delta
    return dataclasses.replace(person, **changes)


# Готові сценарії для сторінки; це демонстрація моделі, не медична рекомендація (DISCLAIMER)
SCENARIO_PRESETS: list[dict] = [
    {"name": "Схуднути до ІМТ 25", "changes": {"bmi_target": 25}},
    {"name": "Талія −10 см", "changes": {"waist_delta": -10}},
    {"name": f"Стати активним ≥ {ACTIVE_MIN_PER_WEEK // 7} хв/день", "changes": {"active": True}},
]


def applicable_presets(person: PersonInput) -> list[dict]:
    """Пресети, застосовні до людини: ІМТ-пресет лише якщо поточний ІМТ > цілі."""
    bmi = bmi_from(person.height_cm, person.weight_kg)
    return [
        s
        for s in SCENARIO_PRESETS
        if "bmi_target" not in s["changes"] or bmi > s["changes"]["bmi_target"]
    ]


def slider_bmi_start(person: PersonInput) -> float:
    """Початкове значення слайдера ІМТ: поточний ІМТ людини в межах слайдера."""
    lo, hi = BMI_SLIDER
    return min(max(round(bmi_from(person.height_cm, person.weight_kg), 1), lo), hi)


def chosen_scenarios(
    person: PersonInput,
    picked: list[str],
    *,
    bmi_target: float,
    waist_delta: float = 0,
    active: bool = False,
) -> list[dict]:
    """Обрані готові сценарії + «Мій сценарій», якщо слайдери чи перемикач зрушено."""
    scenarios = [s for s in applicable_presets(person) if s["name"] in picked]
    changes: dict[str, float | bool] = {}
    if abs(bmi_target - slider_bmi_start(person)) >= BMI_STEP / 2:
        changes["bmi_target"] = bmi_target
    if waist_delta:
        changes["waist_delta"] = waist_delta
    if active:
        changes["active"] = True
    if changes:
        scenarios.append({"name": "Мій сценарій", "changes": changes})
    return scenarios


def format_scenarios(table: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    """Таблиця сценаріїв для показу: «2.0 %» і «−0.2 п.п.» у всіх клітинках."""
    out = table.copy()
    out[model_keys] = table[model_keys].map(interpret.fmt_pct)
    out[CHANGE_COL] = table[CHANGE_COL].map(interpret.fmt_pp)
    return out


def readable_explanation(expl: shap.Explanation) -> shap.Explanation:
    """Explanation для рисунка: бінарні ознаки як «так/ні», числа округлені, назви українською."""
    shown = [
        interpret.feature_display(n, v) for n, v in zip(expl.feature_names, expl.data, strict=True)
    ]
    return shap.Explanation(
        values=expl.values,
        base_values=expl.base_values,
        data=np.array([v for v, _ in shown], dtype=object),
        feature_names=[n for _, n in shown],
    )
