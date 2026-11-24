"""Сторінка 7: форма людини → ймовірності моделей, FINDRISC, SHAP, сценарії «що якщо»."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import shap
import streamlit as st
from sklearn.pipeline import Pipeline

from _common import (
    Numbering,
    disclaimer,
    figure_block,
    fmt_int,
    models_or_stop,
    nhanes_only_banner,
    page_header,
    setup_page,
    table_block,
)
from risk import interpret
from risk.config import (
    ACTIVE_MIN_PER_WEEK,
    DISCLAIMER,
    FEATURE_LABELS_UK,
    FEATURES,
    MODEL_NAMES_UK,
)
from risk.models.explain import local_shap, make_explainer, top_contributions
from risk.person import (
    BMI_SLIDER,
    BMI_STEP,
    CHANGE_COL,
    PersonInput,
    applicable_presets,
    bmi_from,
    chosen_scenarios,
    field_range,
    findrisc_result,
    format_scenarios,
    predict_all,
    readable_explanation,
    scenario_table,
    slider_bmi_start,
    to_feature_row,
    train_medians,
    validate,
)
from risk.viz import figures


@st.cache_resource(show_spinner="Обчислення…")
def get_medians(_models: dict) -> dict[str, float]:
    """Медіани навчальної вибірки; префікс «_» — Streamlit не хешує моделі для кешу."""
    return train_medians(_models)


@st.cache_resource(show_spinner="Обчислення…")
def get_explainer(_pipeline: Pipeline) -> shap.Explainer:
    """TreeExplainer один на сесію сервера; «_» — пайплайн не хешується для кешу."""
    return make_explainer(_pipeline)


LAB_KEYS = ("sbp", "dbp", "tc", "hdl")  # поля з прапорцем «не знаю»


def number(
    label: str, key: str, default: float, note: str = "", disabled: bool = False
) -> float | None:
    """Поле без вбудованих меж: порожнє → None, межі й повідомлення дає validate().

    Тип поля — за типом ``default``: ціле (вік, тиск, активність) або дробове.
    """
    st.session_state.setdefault(key, default)
    whole = isinstance(default, int)
    lo, hi = field_range(key)
    bounds = f"від {fmt_int(int(lo))} до {fmt_int(int(hi))}"
    return st.number_input(
        label,
        value=None,
        step=1 if whole else 1.0,
        format=None if whole else "%g",
        key=key,
        placeholder=bounds,
        help=bounds + ("; ціле число" if whole else "") + note,
        disabled=disabled,
    )


def optional_number(label: str, key: str, default: float) -> float | None:
    """Поле з прапорцем «не знаю»: прапорець вимикає поле, а значення стає None."""
    c1, c2 = st.columns([3, 1])
    unknown = c2.checkbox("не знаю", key=f"{key}_unknown")
    with c1:
        value = number(label, key, default, disabled=unknown)
    return None if unknown else value


def empty_lab_fields() -> list[str]:
    """Поля тиску й холестерину, які лишили порожніми без «не знаю»."""
    state = st.session_state
    return [k for k in LAB_KEYS if state[k] is None and not state[f"{k}_unknown"]]


def activity_input() -> float | None:
    """Фізична активність, хв/тиждень; у підказці — межі поля і норма активності на день."""
    return number(
        "Фізична активність, хв/тиждень",
        "pa_min_week",
        150,
        note=f"; ≥ {ACTIVE_MIN_PER_WEEK} хв/тиждень ≈ {ACTIVE_MIN_PER_WEEK // 7} хв/день",
    )


def lab_inputs() -> tuple[float | None, float | None, float | None, float | None]:
    """Тиск і холестерин (САТ, ДАТ, загальний, ЛПВЩ) з «не знаю» і підказка про медіани."""
    values = (
        optional_number("Систолічний тиск (САТ), мм рт. ст.", "sbp", 120),
        optional_number("Діастолічний тиск (ДАТ), мм рт. ст.", "dbp", 80),
        optional_number("Загальний холестерин, мг/дл", "tc", 190.0),
        optional_number("ЛПВЩ-холестерин, мг/дл", "hdl", 50.0),
    )
    st.caption(
        "Якщо не знаєте значення, відмітьте «не знаю»: усі моделі й пояснення використають "
        "типове значення (медіану навчальної вибірки)."
    )
    return values


def read_form() -> PersonInput:
    """Поля форми; повертає введені значення (без перевірки)."""
    c1, c2 = st.columns(2)
    with c1:
        age = number("Вік, років", "age", 45)
        male = st.radio("Стать", ["Жінка", "Чоловік"], horizontal=True) == "Чоловік"
        height = number("Зріст, см", "height_cm", 170.0)
        weight = number("Вага, кг", "weight_kg", 75.0)
        waist = number("Окружність талії, см", "waist", 90.0)
        pa = activity_input()
    with c2:
        sbp, dbp, tc, hdl = lab_inputs()
        fam = st.checkbox("Цукровий діабет у близьких родичів")
        bp_med = st.checkbox("Приймаю ліки від тиску")
        glu = st.checkbox("Лікарі казали про підвищений цукор / предіабет")
    return PersonInput(age=age, male=male, height_cm=height, weight_kg=weight, waist_cm=waist,
                       sbp=sbp, dbp=dbp, tc=tc, hdl=hdl, pa_min_week=pa, fam_dm=fam,
                       bp_med=bp_med, high_glu_hist=glu)  # fmt: skip


def show_probabilities(models: dict, metrics: dict, person: PersonInput) -> None:
    """Ймовірність від кожної моделі, положення відносно її порогу Юдена і пояснення порогу."""
    st.header("Ймовірності за моделями")
    best, thr = metrics["best_model"], metrics["thresholds"]
    probs = predict_all(models, person, medians=get_medians(models))
    for col, (name, p) in zip(st.columns(len(probs)), probs.items(), strict=True):
        title = MODEL_NAMES_UK[name] + (" (основна)" if name == best else "")
        side = "вище" if p >= thr[name] else "нижче"
        with col:
            st.metric(title, f"{p * 100:.1f} %")
            st.caption(f"{side} порогу ({thr[name] * 100:.1f} %)")
    t = metrics["test"][best]
    st.write(interpret.threshold_text(thr[best], t["recall"], t["specificity"]))
    st.caption(
        "Це ймовірність наявного (можливо, ще не виявленого) діабету 2 типу за даними "
        "обстеження, а не прогноз на 10 років, як в оригінальному FINDRISC."
    )
    st.caption(
        "Поріг — це значення, вище якого модель відносить людину до групи підвищеного ризику."
    )


def show_findrisc(person: PersonInput) -> None:
    """Бал повної шкали FINDRISC (усі 7 пунктів) і категорія ризику."""
    st.header("Шкала FINDRISC")
    score, cat = findrisc_result(person)
    st.markdown(f"<h2>{score} / 25 — ризик {cat}</h2>", unsafe_allow_html=True)
    st.caption(
        "Тут розраховано повний варіант шкали з усіма сімома пунктами. На даних NHANES шкалу "
        "оцінювали у двох варіантах: без пункту про підвищений цукор (a) і лише серед людей "
        "без діагностованого діабету (b), бо в діагностованих цей пункт «підказує» відповідь."
    )


def show_shap(models: dict, person: PersonInput, num: Numbering) -> None:
    """Waterfall-рисунок і текст про внесок ознак у прогноз Gradient Boosting."""
    st.header("Що найбільше вплинуло на результат")
    medians = get_medians(models)
    row = to_feature_row(person, medians=medians)
    expl = local_shap(get_explainer(models["hgb"]), row)
    prob = predict_all({"hgb": models["hgb"]}, person, medians=medians)["hgb"]
    figure_block(
        figures.shap_waterfall,
        (readable_explanation(expl),),
        num.figure(),
        "Внесок ознак у прогноз моделі Gradient Boosting "
        f"(її ймовірність для введених даних — {prob * 100:.1f} %)",
        f"{interpret.SHAP_AXES_NOTE} Червоні стовпчики підвищують оцінку, сині знижують; "
        "довжина показує силу впливу (у log-odds).",
        interpret.shap_text(
            top_contributions(expl, len(FEATURES)), FEATURE_LABELS_UK, row.iloc[0].to_dict()
        ),
    )
    st.caption(CYCLE_NOTE)


CYCLE_NOTE = (
    "Рядок «Період обстеження» — не ваш показник, ви його не вводили. Дані NHANES зібрано у двох "
    "періодах (2015–2016 і 2017–2020), і тиск у них вимірювали різними приладами, тому модель "
    "враховує період. Для нової людини його зафіксовано як «2017–2020» (найближчий до "
    "сьогодення); це припущення, однакове для всіх, а внесок цього рядка входить у загальну суму."
)
LOWER_AT_HIGHER_NOTE = (
    "За деякими показниками (наприклад, діастолічний тиск, загальний холестерин) модель може "
    "давати нижчу оцінку при вищому значенні. Це відображає зв'язки в даних NHANES, зокрема "
    "лікування після встановленого діагнозу, а не користь від високого тиску чи холестерину."
)


def scenario_controls(person: PersonInput) -> list[dict]:
    """Вибір готових сценаріїв і власний сценарій зі слайдерів/перемикача."""
    names = [s["name"] for s in applicable_presets(person)]
    key = hash(person)  # нова людина → елементи скидаються до її значень
    picked = st.multiselect(
        "Готові сценарії", names, default=names, key=f"presets{key}", placeholder="Оберіть сценарії"
    )
    c1, c2 = st.columns(2)
    target = c1.slider(
        "Цільовий ІМТ",
        *BMI_SLIDER,
        slider_bmi_start(person),
        BMI_STEP,
        format="%.1f",
        key=f"bmi{key}",
    )
    delta = c2.slider("Зміна талії, см", -30, 30, 0, key=f"waist{key}")
    active = c1.toggle(f"Активність ≥ {ACTIVE_MIN_PER_WEEK // 7} хв/день", key=f"act{key}")
    return chosen_scenarios(person, picked, bmi_target=target, waist_delta=delta, active=active)


def pp_note(table: pd.DataFrame, best: str) -> None:
    """Пояснення «п.п.» на прикладі першого сценарного рядка таблиці."""
    if len(table) < 2:
        return
    base, r = table[best].iloc[0], table.iloc[1]
    st.caption(
        f"п.п. — відсоткові пункти: було {interpret.fmt_pct(base)}, стало "
        f"{interpret.fmt_pct(r[best])} → {interpret.fmt_pp(r[CHANGE_COL])} ({r['Сценарій']})."
    )


def show_scenarios(models: dict, metrics: dict, person: PersonInput, num: Numbering) -> None:
    """Таблиця «що якщо» і зміна ймовірності основної моделі в кожному сценарії."""
    st.header("Що якщо")
    st.warning(
        "Сценарії показують, як змінюється відповідь моделі, а не що робити. Це демонстрація "
        "поведінки моделі, а не медична порада."
    )
    st.caption(LOWER_AT_HIGHER_NOTE)
    best = metrics["best_model"]
    table = scenario_table(
        models, person, scenario_controls(person), best, medians=get_medians(models)
    )
    skipped = table.attrs.get("skipped")
    if skipped:
        st.caption("Пропущено сценарії з неприпустимими значеннями: " + ", ".join(skipped) + ".")
    show_scenario_table(table, list(models), num)
    pp_note(table, best)
    show_scenario_changes(table, best)


def show_scenario_table(table: pd.DataFrame, model_keys: list[str], num: Numbering) -> None:
    """Таблиця сценаріїв: ймовірності моделей у % і зміна основної моделі в п.п."""
    shown = format_scenarios(table, model_keys).rename(columns=MODEL_NAMES_UK)
    table_block(
        shown,
        num.table(),
        "Ймовірність ЦД2 за моделями в різних сценаріях, %",
        "Перший рядок — ваші введені дані. В інших рядках змінено лише зазначене, решта "
        "лишається як є. Остання колонка — зміна ймовірності основної моделі "
        "у відсоткових пунктах (п.п.).",
        "Мінус означає, що модель оцінює ризик нижче, плюс — вище. Це поведінка моделі на "
        "даних NHANES, а не доведений ефект втручання.",
    )


def show_scenario_changes(table: pd.DataFrame, best: str) -> None:
    """Речення на кожен сценарій: як змінюється ймовірність за основною моделлю."""
    name = MODEL_NAMES_UK[best]
    base = interpret.fmt_pct(table[best].iloc[0])
    for _, r in table.iloc[1:].iterrows():
        st.write(
            f"{r['Сценарій']}: ймовірність за {name} змінюється з {base} "
            f"до {interpret.fmt_pct(r[best])}."
        )


def show_result(models: dict, metrics: dict, candidate: PersonInput, submitted: bool) -> None:
    """Результат для людини, що була оцінена кнопкою; зміни у формі ховають старий результат."""
    if errors := validate(candidate, empty_lab_fields()):
        for msg in errors:
            st.error(msg)
        return
    if submitted:
        st.session_state["person"] = candidate
    person = st.session_state.get("person")
    if person is None:
        return
    if person != candidate:
        st.warning("Дані змінено — натисніть «Оцінити ризик»")
        return
    num = Numbering(7)
    st.info(f"Ваш ІМТ: {bmi_from(person.height_cm, person.weight_kg):.1f} кг/м²")
    show_probabilities(models, metrics, person)
    show_findrisc(person)
    show_shap(models, person, num)
    show_scenarios(models, metrics, person, num)


setup_page("Оцінка ризику", "👤")
page_header(
    "Оцінка ризику",
    "Введіть свої дані, і три моделі оцінять ймовірність діабету 2 типу, а шкала FINDRISC "
    "дасть бали за анкетою. Далі видно, які чинники найбільше вплинули, і можна подивитися, "
    "як змінилася б оцінка моделі за інших значень.",
)
nhanes_only_banner("Ця сторінка не залежить від вашого файлу: моделі навчені на NHANES.")
st.warning(DISCLAIMER)
st.caption(
    "Сценарії нижче — це демонстрація поведінки моделі, а не медична порада. "
    "Дані не зберігаються й нікуди не надсилаються."
)

models, metrics = models_or_stop()

candidate = read_form()
submitted = st.button("Оцінити ризик")
show_result(models, metrics, candidate, submitted)

disclaimer()
