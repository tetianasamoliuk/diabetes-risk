"""Прийняття/відхилення файла, дисклеймери на всіх станах, сценарій «Мій сценарій»."""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from risk.config import DISCLAIMER
from risk.data.user_data import validate_user_file
from risk.person import PersonInput, findrisc_result
from tests.app.conftest import user_csv

PAGES = Path(__file__).resolve().parents[2] / "app" / "pages"
# AppTest виконує скрипт у цьому ж процесі: `from _common import …` потребує app/ у sys.path
sys.path.insert(0, str(PAGES.parent))
P4, P5, P7 = (next(PAGES.glob(f"{n}_*.py")) for n in "457")


def _has_disclaimer(at: AppTest) -> bool:
    return any(DISCLAIMER in e.value for e in [*at.caption, *at.markdown, *at.warning, *at.info])


def _accept_app() -> None:
    import streamlit as st

    from _common import accept_user_file

    ok = accept_user_file(st.session_state["ok"], "a.csv", "1")
    st.session_state["after_ok"] = (list(ok), "user_df" in st.session_state)
    bad = accept_user_file(b"text", "a.txt", "2")
    st.session_state["after_bad"] = (list(bad), "user_df" in st.session_state)


def test_rejected_file_clears_accepted() -> None:
    at = AppTest.from_function(_accept_app, default_timeout=60)
    at.session_state["ok"] = user_csv()
    at.run()
    assert not at.exception
    assert at.session_state["after_ok"] == ([], True)
    errors, has_df = at.session_state["after_bad"]
    assert errors and not has_df


def test_pages_4_5_disclaimer_on_user_data() -> None:
    user_df, _ = validate_user_file(user_csv(), "x.csv")
    for page in (P4, P5):
        at = AppTest.from_file(str(page), default_timeout=240)
        at.session_state["user_df"] = user_df
        at.run()
        assert not at.exception
        assert _has_disclaimer(at), page.name


def _submit(at: AppTest) -> AppTest:
    return at.button[0].click().run()


def test_page_7_disclaimer_after_submit_and_errors() -> None:
    at = AppTest.from_file(str(P7), default_timeout=240).run()
    assert _has_disclaimer(_submit(at))
    at.number_input[1].set_value(250)  # зріст 250 + вага 30 → некоректний ІМТ
    at.number_input[2].set_value(30)
    _submit(at)
    assert at.error and _has_disclaimer(at)


def test_page_7_values_and_no_auto_scenario_for_bmi_48() -> None:
    at = AppTest.from_file(str(P7), default_timeout=240).run()
    at.number_input[1].set_value(170)
    at.number_input[2].set_value(139)  # ІМТ ≈ 48
    _submit(at)
    assert not at.exception
    values = [float(m.value.rstrip(" %").replace(",", ".")) for m in at.metric]
    assert len(values) == 3 and all(0 <= v <= 100 for v in values)
    table = next(d.value for d in at.dataframe if "Сценарій" in d.value.columns)
    assert "Мій сценарій" not in set(table["Сценарій"])


def test_page_7_findrisc_matches_person_result() -> None:
    at = _submit(AppTest.from_file(str(P7), default_timeout=240).run())
    person = PersonInput(45, False, 170, 75, 90, 120, 80, 190, 50, 150, False, False, False)
    score, cat = findrisc_result(person)
    assert any(f"{score} / 25 — ризик {cat}" in m.value for m in at.markdown)
