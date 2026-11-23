"""Сторінка 8: Excel з усіма таблицями і графіками та ZIP із рисунками."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from _common import (
    Numbering,
    disclaimer,
    get_models,
    load_cohort,
    load_flow,
    load_variables,
    models_or_stop,
    nhanes_only_banner,
    page_header,
    setup_page,
    table_block,
)
from risk.external.ukraine import load_ukraine_stats
from risk.models.train import load_test_predictions
from risk.report.results import (
    ResultsBundle,
    build_results,
    export_excel,
    export_zip,
    to_excel_sheets,
)

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@st.cache_resource(show_spinner="Збираємо таблиці й графіки…")
def build() -> ResultsBundle:
    """Усі таблиці, рисунки й числа на NHANES; збираються один раз за сесію сервера."""
    models, metrics = get_models()
    return build_results(
        load_cohort(),
        load_flow(),
        load_variables(),
        models=models,
        metrics=metrics,
        preds=load_test_predictions(),
        ukraine=load_ukraine_stats(),
    )


@st.cache_data(show_spinner="Підготовка файла…")
def excel_bytes() -> bytes:
    """Книга .xlsx для кнопки завантаження."""
    return export_excel(build())


@st.cache_data(show_spinner="Підготовка файла…")
def zip_bytes() -> bytes:
    """ZIP з PNG усіх рисунків для кнопки завантаження."""
    return export_zip(build())


setup_page("Експорт", "📥")
page_header(
    "Експорт",
    "Одна кнопка і всі таблиці та графіки опиняються у файлі Excel, щоб вставити їх у текст "
    "роботи. Окремо можна забрати всі графіки архівом.",
)
nhanes_only_banner(
    "Експорт завжди робиться на даних NHANES. Результати для вашого файлу можна "
    "завантажити кнопками PNG на відповідних сторінках."
)

models_or_stop()
bundle = build()
st.header("Що потрапить у файл")
st.markdown("\n".join(f"- {s.name}" for s in to_excel_sheets(bundle)) + "\n- Графіки\n- Пояснення")
st.download_button(
    "Завантажити Excel з усіма результатами",
    data=excel_bytes(),
    file_name="результати_ЦД2.xlsx",
    mime=XLSX_MIME,
)
st.download_button(
    "Завантажити всі графіки (ZIP)",
    data=zip_bytes(),
    file_name="графіки_ЦД2.zip",
    mime="application/zip",
)

st.header("Пояснення термінів")
table_block(
    bundle.glossary,
    Numbering(8).table(),
    "Глосарій термінів",
    "Ліворуч термін, праворуч пояснення простою мовою.",
    "Той самий глосарій потрапляє в Excel на лист «Пояснення».",
)
disclaimer()
