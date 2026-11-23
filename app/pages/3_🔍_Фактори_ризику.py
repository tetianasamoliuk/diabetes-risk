"""Сторінка 3: гіпотези, відношення шансів, ранжування факторів."""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from _common import (
    NO_VARIATION_MSG,
    Numbering,
    active_dataset,
    available,
    dataset_banner,
    disclaimer,
    figure_block,
    fmt_int,
    page_header,
    setup_page,
    table_block,
)
from risk import interpret
from risk.config import (
    BINARY_FACTORS,
    CONTINUOUS_FACTORS,
    FEATURE_LABELS_UK,
    FEATURES,
    OR_SCALE_LABELS_UK,
)
from risk.data.user_data import complete_cases_shortfall
from risk.errors import DataValidationError
from risk.report.results import FACTORS, OR_COLS_UK, share_cells
from risk.stats.hypothesis import run_all_tests
from risk.stats.odds import (
    clinical_or_table,
    multivariable_logit,
    rank_factors,
    univariate_or,
)
from risk.viz import figures


@st.cache_data(show_spinner="Обчислення…")
def compute_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Тести порівняння груп лише для ознак, що мають різні значення в наборі."""
    cont = available(df, CONTINUOUS_FACTORS)
    cat = available(df, [c for c in BINARY_FACTORS if c != "cycle_2017"])
    return run_all_tests(df, cont, cat)


@st.cache_data(show_spinner="Обчислення…")
def compute_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """Список OddsRatio у таблицю з індексом-змінною."""
    rows = [asdict(univariate_or(df, f)) for f in available(df, FACTORS)]
    return pd.DataFrame(rows).set_index("variable")


@st.cache_data(show_spinner="Обчислення…")
def compute_multivariable(df: pd.DataFrame) -> pd.DataFrame | str:
    """Багатофакторна модель або текст, чому її не побудовано."""
    features = available(df, FEATURES)
    if shortfall := complete_cases_shortfall(df, features):
        return shortfall
    try:
        t = multivariable_logit(df, features)
    except DataValidationError as e:
        return str(e)
    t.attrs = {"n": t.attrs["n"], "converged": t.attrs["converged"]}
    return t


@st.cache_data(show_spinner="Обчислення…")
def compute_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Кешована clinical_or_table для активного набору."""
    return clinical_or_table(df)


def fmt_table(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Українські назви змінних, p у форматі «< 0.001», решта до 2 знаків, «—» замість nan."""
    out = t[cols].apply(lambda c: c.map(lambda v: interpret.fmt_cell(v, cap=1000)))
    out["p"] = t["p"].map(interpret.fmt_p)
    out.insert(0, "Фактор", [FEATURE_LABELS_UK.get(i, i) for i in out.index])
    return out.reset_index(drop=True)


def show_multivariable(multi: pd.DataFrame, n_total: int, num: Numbering) -> None:
    """Багатофакторна модель, форест-графік, VIF і ранжування."""
    show_multivariable_table(multi, n_total, num)
    if multi.attrs["converged"]:
        show_multivariable_forest(multi, num)
    st.markdown(interpret.vif_text(multi))
    show_ranking(multi, num)


def show_multivariable_table(multi: pd.DataFrame, n_total: int, num: Numbering) -> None:
    """Скільки людей у моделі, попередження про збіжність і таблиця коефіцієнтів з VIF."""
    mt = multi.drop(index="const")
    mt_view = fmt_table(mt, ["coef", "odds_ratio", "ci_low", "ci_high", "p", "vif"])
    mt_view.columns = ["Фактор", "Коефіцієнт", "OR", "95% ДІ від", "95% ДІ до", "p", "VIF"]
    st.write(
        f"Розраховано на повних випадках аналітичної вибірки (N = {fmt_int(multi.attrs['n'])} "
        f"з {fmt_int(n_total)}). "
        "Це опис зв'язків, а не прогноз, тому поділу на навчальну й тестову частини тут немає."
    )
    if not multi.attrs["converged"]:
        st.warning("Модель не збіглася: оцінки можуть бути ненадійними.")
    table_block(
        mt_view,
        num.table(),
        "Багатофакторна логістична регресія (усі фактори разом)",
        "Тут усі фактори враховані одночасно: OR кожного показує його вплив за умови, що "
        "решта незмінна. VIF вимірює, наскільки фактор дублює інші.",
        "Багатофакторний OR може відрізнятися від однофакторного, бо фактори пов'язані між "
        "собою. Це нормально і не є помилкою.",
    )


def show_multivariable_forest(multi: pd.DataFrame, num: Numbering) -> None:
    """Форест-графік багатофакторних OR (без індикатора циклу)."""
    figure_block(
        figures.forest_plot(
            multi.drop(index="cycle_2017", errors="ignore"), "Багатофакторні OR з 95% ДІ"
        ),
        num.figure(),
        "Форест-графік (forest plot) багатофакторних відношень шансів",
        "Як і раніше: точка показує OR, лінії є 95% ДІ, пунктир позначає OR = 1, шкала "
        "логарифмічна.",
        "Тут видно внесок кожного фактора з поправкою на інші. Індикатор циклу входить "
        "у модель як технічна поправка і не показується як фактор.",
    )


def show_ranking(multi: pd.DataFrame, num: Numbering) -> None:
    """Ранжування факторів за стандартизованим ефектом."""
    st.header("Ранжування факторів")
    ranked = rank_factors(multi)
    rk = fmt_table(ranked, ["std_effect", "odds_ratio", "ci_low", "ci_high", "p"])
    rk.columns = ["Фактор", "Стандартизований ефект", "OR", "95% ДІ від", "95% ДІ до", "p"]
    rk.insert(0, "Місце", ranked["rank"].to_numpy())
    table_block(
        rk,
        num.table(),
        "Ранжування факторів за багатофакторною моделлю",
        "Фактори впорядковано за стандартизованим ефектом: |коефіцієнт| × стандартне "
        "відхилення ознаки. Так зв'язки різних ознак (роки, см, так/ні) можна порівнювати. "
        "Значення p показано окремо.",
        "Верхні рядки — це фактори з найсильнішим зв'язком з діабетом. Сила зв'язку не те саме, "
        "що його надійність: її показує p. Це опис зв'язку, а не доказ причини. "
        "Індикатор циклу входить у модель як технічна поправка і не показується як фактор.",
    )


setup_page("Фактори ризику", "🔍")
page_header(
    "Фактори ризику",
    "Тут перевіряється, які ознаки справді пов'язані з діабетом 2 типу. Також видно, "
    "для яких ознак різниця між групами могла виникнути випадково. Спочатку кожну ознаку "
    "порівнюють між групами, потім "
    "рахують, у скільки разів вона збільшує шанси на діабет, окремо та разом з іншими. "
    "Наприкінці ознаки ранжуються за силою зв'язку.",
)

df, is_user = active_dataset()
dataset_banner(is_user, len(df))
num = Numbering(3)

st.header("Гіпотези та статистичні тести")
tests = compute_tests(df)
if tests.empty:
    st.info(NO_VARIATION_MSG)
    disclaimer()
    st.stop()
st.markdown(
    "H0 стверджує, що різниці між групами немає, а H1, що вона є. Якщо дані "
    "суперечать H0, ми її відхиляємо. Для числових ознак формулювання залежить від тесту: "
    "критерій Манна–Вітні порівнює розподіли, t-критерій Велча середні.\n\n"
    + interpret.TEST_CHOICE_NOTE
)
table_block(
    tests[["label", "h0_text", "h1_text"]].rename(
        columns={"label": "Фактор", "h0_text": "H0", "h1_text": "H1"}
    ),
    num.table(),
    "Нульова та альтернативна гіпотези для кожного фактора",
    "Кожен рядок містить пару гіпотез для одного фактора.",
    "Формулювання відповідає обраному тесту.",
    wrap=True,
)

shown = pd.DataFrame(
    {
        "Фактор": tests["label"],
        "Тест": tests["test"].map(interpret.test_name_uk),
        "Статистика": tests["statistic"].round(2),
        "p": tests["p"].map(interpret.fmt_p),
        "p (Холм)": tests["p_holm"].map(interpret.fmt_p),
        **share_cells(tests),
        "Висновок": tests["reject_h0"].map({True: "відхиляємо H0", False: "не відхиляємо H0"}),
    }
)
table_block(
    shown,
    num.table(),
    "Порівняння груп без ЦД2 і з ЦД2: тести та висновки",
    "Для кожного фактора вказано тест, його статистику, p-значення (показує, наскільки "
    "ймовірно отримати таку або більшу різницю, якби насправді її не було) "
    "і p після поправки Холма на множинні порівняння. Для ознак «так/ні» (стать, діабет у "
    "родичів, ліки від тиску) додано частку людей з ознакою в кожній групі; для числових "
    "ознак описова статистика — на сторінці «Статистика».",
    "Висновок «відхиляємо H0» означає, що різниця між групами статистично значуща. "
    "Поправка потрібна, бо при багатьох тестах деякі «значущі» результати виникають випадково.",
    wrap=True,
)
with st.expander("Пояснення до кожного тесту"):
    for r in tests.itertuples():
        st.write(interpret.test_text(r.label, r.test, r.statistic, r.p, r.p_holm))

st.header("Однофакторні відношення шансів")
uni = compute_univariate(df)
uni_view = fmt_table(uni, ["odds_ratio", "ci_low", "ci_high", "p"]).rename(columns=OR_COLS_UK)
uni_view["Фактор"] = uni["label"].to_numpy()
uni_view["N"] = uni["n"].to_numpy()
table_block(
    uni_view.reset_index(drop=True),
    num.table(),
    "Однофакторні відношення шансів (кожен фактор окремо)",
    "OR показує, у скільки разів відрізняються шанси мати діабет. Для так/ні-ознак "
    "порівнюються люди з ознакою і без неї; для числових, зростання на вказаний у назві крок "
    "(наприклад, 10 років). ДІ — довірчий інтервал.",
    "OR більше 1 означає вищі шанси, менше 1 нижчі. Якщо інтервал не містить 1, зв'язок "
    "статистично значущий. Кроки для числових ознак різні, тому їхні OR між собою "
    "напряму не порівнюють.",
)
figure_block(
    figures.forest_plot(uni, "Однофакторні OR з 95% ДІ"),
    num.figure(),
    "Форест-графік (forest plot) однофакторних відношень шансів",
    "Кожна точка показує OR, лінії навколо неї 95% довірчий інтервал. Пунктир позначає "
    "OR = 1, тобто відсутність зв'язку. Шкала логарифмічна.",
    "Інтервали, що не перетинають пунктир, відповідають значущим факторам. "
    "Для числових ознак OR розраховано на крок з таблиці вище.",
)


def uni_text(name: str, r: pd.Series) -> str:
    """Пояснення OR: для числових із кроком, для так/ні з референсом «у решти»."""
    ci = (r["odds_ratio"], r["ci_low"], r["ci_high"])
    if name in BINARY_FACTORS:
        return interpret.or_text(FEATURE_LABELS_UK[name].lower(), *ci, "у решти")
    step = OR_SCALE_LABELS_UK[name].removeprefix("на ")
    return interpret.or_text(FEATURE_LABELS_UK[name].split(",")[0].lower(), *ci, scale_label=step)


with st.expander("Пояснення до кожного OR"):
    for name, r in uni.dropna(subset=["odds_ratio", "ci_low", "ci_high"]).iterrows():
        st.write(uni_text(name, r))

st.header("OR за клінічними порогами")
clin = compute_clinical(df)
clin_view = clin.round({"odds_ratio": 2, "ci_low": 2, "ci_high": 2}).assign(
    p=clin["p"].map(interpret.fmt_p)
)
clin_view.columns = ["Поріг", "OR", "95% ДІ від", "95% ДІ до", "p", "N із ознакою", "N"]
table_block(
    clin_view,
    num.table(),
    "Відношення шансів за клінічними порогами",
    "Люди з ознакою (за порогом) порівнюються з тими, хто нижче порогу. Для талії поріг "
    "залежить від статі, для активності ознакою є значення нижче порогу.",
    "Такі OR зрозуміліші в клініці, ніж OR на крок, бо відповідають звичним критеріям ризику.",
)
with st.expander("Пояснення до порогових OR"):
    for r in clin.itertuples():
        st.write(
            interpret.or_text(r.label, r.odds_ratio, r.ci_low, r.ci_high, "у людей нижче порогу")
        )

st.header("Багатофакторна логістична регресія")
multi = compute_multivariable(df)
if isinstance(multi, str):
    st.info(multi)
else:
    show_multivariable(multi, len(df), num)

disclaimer()
