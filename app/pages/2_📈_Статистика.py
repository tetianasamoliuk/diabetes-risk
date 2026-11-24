"""Сторінка 2: описова статистика, розподіли, кореляції, поширеність."""

import sys
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
    group_name,
    has_survey_design,
    page_header,
    prevalence_table,
    setup_page,
    table_block,
)
from risk import interpret
from risk.config import CONTINUOUS_FACTORS, FEATURE_LABELS_UK
from risk.report.results import BOX_FACTORS, CORR_METHODS, DESCRIBE_COLS_UK, prevalence_view
from risk.stats.correlation import corr_matrix
from risk.stats.descriptive import describe_by_group
from risk.viz import figures


@st.cache_data(show_spinner="Обчислення…")
def describe_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Описова статистика з українськими назвами змінних, груп і колонок."""
    t = describe_by_group(df, cols).reset_index()
    t["variable"] = t["variable"].map(FEATURE_LABELS_UK)
    t["group"] = t["group"].map(group_name)
    return t.rename(columns=DESCRIBE_COLS_UK).round(2)


@st.cache_data(show_spinner="Обчислення…")
def correlations(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    """Матриці кореляцій Пірсона і Спірмена."""
    return {m: corr_matrix(df, cols, m) for m, _ in CORR_METHODS}


setup_page("Статистика", "📈")
page_header(
    "Статистика",
    "Тут ми порівнюємо людей з діабетом 2 типу і без нього за основними ознаками: віком, "
    "вагою, тиском, холестерином. Таблиці й графіки показують, наскільки ці групи "
    "відрізняються. Окремо видно, наскільки ознаки пов'язані між собою і яка частка людей "
    "хворіє на діабет.",
)

df, is_user = active_dataset()
dataset_banner(is_user, len(df))
num = Numbering(2)
cols = available(df, CONTINUOUS_FACTORS)

st.header("Описова статистика")
if not cols:
    st.info(NO_VARIATION_MSG)
    disclaimer()
    st.stop()
table_block(
    describe_table(df, cols),
    num.table(),
    "Описова статистика неперервних ознак за групами",
    "Для кожної ознаки та групи: кількість людей (N), середнє і SD (розкид навколо середнього), "
    "медіана, квартилі Q1 і Q3, міжквартильний розмах IQR, мінімум і максимум.",
    "Якщо середнє і медіана сильно різняться, розподіл несиметричний. Порівняння середніх у "
    "двох групах дає перше уявлення, які ознаки відрізняють людей з діабетом.",
)

st.header("Розподіли ознак")
chosen = st.selectbox(
    "Ознака для гістограми", cols, format_func=FEATURE_LABELS_UK.get, placeholder="Оберіть ознаку"
)
figure_block(
    figures.hist_by_group,
    (df, chosen, FEATURE_LABELS_UK[chosen]),
    num.figure(),
    f"Розподіл ознаки за групами: {FEATURE_LABELS_UK[chosen]}",
    "По горизонталі значення ознаки, по вертикалі щільність (де людей більше, там стовпчик "
    "вищий). Дві групи накладено одна на одну.",
    "Якщо розподіли зсунуті відносно один одного, ознака відрізняє групи; якщо майже "
    "збігаються, ознака майже не допомагає.",
)

box_cols = available(df, BOX_FACTORS)
figure_block(
    figures.box_by_group,
    (df, box_cols, [FEATURE_LABELS_UK[c] for c in box_cols]),
    num.figure(),
    "Діаграми розмаху (boxplot) ключових ознак за групами",
    "Прямокутник охоплює середні 50% людей, лінія всередині — це медіана, «вусики» показують "
    "типовий діапазон, окремі точки — це незвичайно великі чи малі значення.",
    "Чим вище прямокутник однієї групи відносно іншої, тим більша типова величина ознаки "
    "в цій групі.",
)

st.header("Кореляції між ознаками")
corr = correlations(df, cols)
for method, name in CORR_METHODS:
    figure_block(
        figures.corr_heatmap,
        (corr[method], f"Кореляційна матриця ({name})"),
        num.figure(),
        f"Матриця кореляцій {name}",
        "Колір і число в клітинці показують силу зв'язку двох ознак від −1 до 1: "
        "біля 0 зв'язку немає, біля 1 вони зростають разом, біля −1 одна зростає, інша спадає.",
        "Сильно пов'язані ознаки (наприклад, індекс маси тіла і талія) несуть схожу "
        "інформацію, це важливо для багатофакторного аналізу. Кореляція Пірсона шукає "
        "лінійний зв'язок, Спірмена працює з рангами і стійкіша до викидів.",
    )
    table_block(
        corr[method]
        .rename(index=FEATURE_LABELS_UK, columns=FEATURE_LABELS_UK)
        .round(3)
        .reset_index(names="Змінна"),
        num.table(),
        f"Матриця кореляцій {name}",
        "Кожна клітинка — коефіцієнт кореляції двох ознак від −1 до 1; ті самі числа, "
        "що на рисунку вище.",
        "Дивіться на клітинки далеко від 0: там зв'язок між ознаками найсильніший.",
        collapsed=True,
    )


st.header("Поширеність діабету")
if is_user or not has_survey_design(df):
    st.info("Зважена поширеність розраховується лише для даних NHANES (потрібні ваги опитування).")
else:
    prev = prevalence_table(df)
    table_block(
        prevalence_view(prev),
        num.table(),
        "Поширеність ЦД2: загалом і за циклами (незважена та зважена)",
        "Незважена частка — це відсоток хворих у самій вибірці. Зважена враховує ваги "
        "опитування, і до неї додано 95% довірчий інтервал (ДІ).",
        "Зважена оцінка показує, яка частка дорослого населення США має діабет, тому саме її "
        "порівнюють з офіційною статистикою.",
    )
    figure_block(
        figures.prevalence_bars,
        (prev,),
        num.figure(),
        "Поширеність ЦД2: незважена та зважена оцінки",
        "Для кожної групи два стовпчики: сірий — це частка у вибірці, синій — це зважена оцінка "
        "з довірчим інтервалом.",
        "Різниця між стовпчиками показує, наскільки вибірка відрізняється від населення.",
    )
    total = prev.iloc[0]
    st.write(
        interpret.prevalence_text(
            total["weighted_pct"], total["ci_low"], total["ci_high"], total["unweighted_pct"]
        )
    )
    st.markdown(
        "**Чому ваги не застосовуються в моделях.** Ваги потрібні, щоб оцінити частку "
        "хворих у населенні. Мета моделей інша: розрізняти ризик окремих людей, а не "
        "оцінювати популяційний показник, тому моделі навчаються без ваг, як і прийнято "
        "в такій задачі."
    )

disclaimer()
