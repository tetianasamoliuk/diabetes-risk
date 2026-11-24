"""Сторінка 6: поширеність діабету в США (NHANES, NCHS) і Україні (IDF, МОЗ)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from _common import (
    Numbering,
    disclaimer,
    figure_block,
    fmt_int,
    has_survey_design,
    load_cohort,
    nhanes_only_banner,
    page_header,
    setup_page,
    table_block,
)
from risk.config import EXTERNAL_DIR
from risk.external.ukraine import (
    CALC_NOTE,
    OTHER_COLS_UK,
    UKRAINE_CSV_COLUMNS,
    load_ukraine_stats,
    load_us_reference,
    moz_idf_text,
    other_indicators,
    prevalence_comparison,
)
from risk.report.results import COMPARISON_CAPTION, comparison_view, dots_data, nhanes_rows
from risk.viz import figures


def load_ukraine() -> pd.DataFrame:
    """Дані по Україні з CSV; шлях — від EXTERNAL_DIR на момент запуску сторінки, без кешу."""
    return load_ukraine_stats(EXTERNAL_DIR / "ukraine_stats.csv")


def _value_text(row: pd.Series) -> str:
    """Відсотки джерела — з одним знаком, кількості осіб — цілі з пробілом тисяч."""
    value = row["Значення"]
    return f"{value:.1f}" if row["Одиниця"] == "%" else fmt_int(round(value))


def show_no_data() -> None:
    """Коли файла з даними по Україні немає: перелік джерел і порожній шаблон."""
    st.info(
        "Першоджерельні дані по Україні не отримані. Нижче перелік перевірених джерел і "
        "порожній шаблон таблиці."
    )
    st.dataframe(pd.DataFrame(columns=UKRAINE_CSV_COLUMNS), width="stretch")


def show_comparison(cohort: pd.DataFrame, ukraine: pd.DataFrame, us: pd.DataFrame) -> None:
    """Таблиця 6.1, рисунок 6.1, висновок про облік МОЗ і таблиця 6.2."""
    num = Numbering(6)
    comparison = prevalence_comparison(nhanes_rows(cohort), ukraine, us)
    show_comparison_table(comparison, num)
    show_comparison_figure(comparison, num)
    if text := moz_idf_text(ukraine):
        st.subheader("Облік МОЗ і оцінка IDF")
        st.success(text)
    show_other_indicators(other_indicators(ukraine, us), num)


def show_comparison_table(comparison: pd.DataFrame, num: Numbering) -> None:
    """Таблиця 6.1: поширеність з різних джерел, згрупована за тим, що вимірюється."""
    table_block(
        comparison_view(comparison),
        num.table(),
        COMPARISON_CAPTION,
        "Рядки згруповано за тим, що саме вимірюється: діабет загалом, діагностований і "
        "недіагностований. Рядки NHANES — ця робота (зважені оцінки з 95% ДІ). Далі офіційні "
        f"оцінки NCHS для США, оцінка IDF для України і рядки з позначкою «{CALC_NOTE}»: це "
        "частки, обчислені з кількостей осіб, які дають IDF і МОЗ, від дорослого населення "
        "20–79 років за IDF. Довірчий інтервал є лише там, де його дає джерело, інакше «—».",
        "Рядки NHANES і NCHS для діагностованого діабету — пряма зовнішня перевірка вибірки: "
        "обидва спираються на самозвіт про діагноз. Загальна й недіагностована поширеність "
        "відрізняються сильніше, бо в цій роботі недіагностований діабет визначається лише за "
        "HbA1c, без глюкози натще, і роки інші. Для України напряму порівнянні рядки IDF і МОЗ "
        "для діагностованого діабету.",
        wrap=True,
    )


def show_comparison_figure(comparison: pd.DataFrame, num: Numbering) -> None:
    """Рисунок 6.1: точки поширеності з довірчими інтервалами там, де вони відомі."""
    figure_block(
        figures.prevalence_dots,
        (dots_data(comparison),),
        num.figure(),
        COMPARISON_CAPTION,
        "Кожна точка — одне джерело; у підписі вказано, що саме вимірюється, а під ним — метод "
        "і вік. Горизонтальний відрізок — довірчий інтервал там, де він відомий.",
        "Порівнювати варто точки з однаковим показником: діагностований діабет за IDF і "
        "за обліком МОЗ в Україні, діагностований діабет NHANES і NCHS у США.",
    )


def show_other_indicators(indicators: pd.DataFrame, num: Numbering) -> None:
    """Таблиця 6.2: вихідні кількості й інші показники; значення — у вигляді джерела."""
    others = indicators.rename(columns=OTHER_COLS_UK)
    table_block(
        others,
        num.table(),
        "Інші показники",
        "Вихідні числа з тих самих джерел: кількості осіб з діабетом і недіагностованим "
        "діабетом за IDF, доросле населення, пацієнти в обліку МОЗ, давніша оцінка IDF. "
        "Адреса джерела і дата перевірки наведені в таблиці.",
        f"З цих кількостей розраховано частки з позначкою «{CALC_NOTE}» у таблиці порівняння.",
        wrap=True,
        shown=others.assign(Значення=others.apply(_value_text, axis=1)),
    )


setup_page("Порівняння з Україною", "🇺🇦")
page_header(
    "Порівняння з Україною",
    "Тут ми ставимо поруч частку дорослих з діабетом у США за даними NHANES і в Україні за "
    "відкритими джерелами. Це допомагає оцінити, наскільки наші висновки з американської "
    "вибірки доречні для українського контексту. Дані по Україні взято з сайтів IDF Diabetes "
    "Atlas і Міністерства охорони здоров'я; для США додано офіційну оцінку NCHS як зовнішню "
    "перевірку. Кожне число має посилання і дату перевірки. Де вимірюється одне й те саме "
    "(наприклад, частка дорослих із діагностованим діабетом), числа порівнюємо напряму; "
    "решта — орієнтир, бо джерела рахують по-різному.",
)
nhanes_only_banner(
    "Ця сторінка завжди показує NHANES: порівняння з Україною не залежить від вашого файлу."
)
st.caption(
    "Порівняння завжди будується на аналітичній вибірці NHANES, а не на завантажених вами даних."
)

ukraine = load_ukraine()
cohort = load_cohort()

st.header("Поширеність: США і Україна")
if ukraine.empty:
    show_no_data()
elif not has_survey_design(cohort):
    st.info("Для NHANES немає ваг опитування, зважену поширеність розрахувати неможливо.")
else:
    show_comparison(cohort, ukraine, load_us_reference())

st.header("Обмеження порівняння")
st.warning(
    "**Обмеження порівняння.** Ці числа не є вимірами одного й того самого.\n\n"
    "- **Різні методи.** NHANES у цій роботі: HbA1c і встановлений діагноз. NCHS: глюкоза "
    "натще або HbA1c і встановлений діагноз. IDF дає модельну оцінку, екстрапольовану з "
    "сусідніх країн, як зазначено на сторінці джерела. Дані МОЗ — облік уже діагностованих "
    "пацієнтів.\n"
    "- **Різні вікові межі.** Ця робота рахує дорослих від 20 років без верхньої межі, NCHS — "
    "від 18 років, IDF бере 20–79 років зі стандартизацією за віком. Облік МОЗ охоплює всі "
    "віки й усі типи діабету, тому його частка від дорослих 20–79 років — наближена.\n"
    "- **Стандартизована і сира частка.** Стандартизована за віком оцінка IDF потрібна для "
    "порівняння країн з різним віковим складом; фактична частка хворих серед дорослих — "
    "сира, її розраховано з кількостей IDF.\n"
    "- **Різні роки.** Оцінки стосуються різних періодів (облік МОЗ — інший рік, ніж оцінка "
    "IDF), а ситуація в Україні могла змінитися.\n"
    "- **Діагностований і недіагностований.** Загальна поширеність NHANES і NCHS включає "
    "людей, які не знають про свій діабет; облік МОЗ їх не бачить, тому МОЗ порівнюємо лише "
    "з діагностованим діабетом."
)
if not ukraine.empty:
    st.markdown("**Джерела:**")
    urls = pd.concat([ukraine["url"], load_us_reference()["url"]]).drop_duplicates()
    st.markdown("\n".join(f"- {u}" for u in urls))

disclaimer()
