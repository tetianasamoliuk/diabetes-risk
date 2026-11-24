"""Smoke-тести сторінок «Дані», «Статистика», «Фактори ризику» через AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from risk.data.user_data import validate_user_file
from risk.interpret import TEST_CHOICE_NOTE
from tests.app.conftest import user_csv

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = [
    APP / "Головна.py",
    APP / "pages" / "1_📊_Дані_і_вибірка.py",
    APP / "pages" / "2_📈_Статистика.py",
    APP / "pages" / "3_🔍_Фактори_ризику.py",
]


def run(path: Path) -> AppTest:
    return AppTest.from_file(str(path), default_timeout=120).run()


def test_data_page_has_title_and_template_button() -> None:
    at = run(PAGES[1])
    assert at.title[0].value == "Дані і вибірка"
    assert any(b.key == "dl_template" for b in at.get("download_button"))


def test_variables_table_wraps_long_text() -> None:
    at = run(PAGES[1])
    # 1.2 — статична таблиця з перенесенням: довгу назву ознаки не обрізає
    assert any("Код NHANES" in t.value.columns for t in at.table)


def test_correlation_tables_fit_page_width() -> None:
    at = run(PAGES[2])
    # 2.2 і 2.3 — статичні: усі дев'ять колонок видно без прокручування вбік
    wide = [t.value for t in at.table if t.value.columns[0] == "Змінна"]
    assert len(wide) == 2
    assert all(len(w.columns) == 9 for w in wide)


def test_risk_factors_page_has_tables() -> None:
    at = run(PAGES[3])
    assert len(at.dataframe) >= 1
    assert TEST_CHOICE_NOTE in " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("path", PAGES[2:], ids=lambda p: p.name)
def test_pages_work_on_user_data(path: Path) -> None:
    user_df, errors = validate_user_file(user_csv(), "x.csv")
    assert user_df is not None, errors
    at = AppTest.from_file(str(path), default_timeout=120)
    at.session_state["user_df"] = user_df
    at.run()
    assert not at.exception
    assert any("завантаженому файлі" in w.value for w in at.warning)


def test_statistics_page_uses_ukrainian_interval_abbreviation() -> None:
    text = " ".join(m.value for m in run(PAGES[2]).markdown)
    assert "довірчий інтервал (ДІ)" in text
    assert "(CI)" not in text
