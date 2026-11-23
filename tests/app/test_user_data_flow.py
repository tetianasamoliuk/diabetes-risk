"""Файл користувача на сторінках: скидання, банер, вироджені набори, таблиці й підписи."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from risk.config import DISCLAIMER
from risk.data.user_data import MIN_PER_CLASS, MIN_ROWS, validate_user_file
from tests.app.conftest import user_csv

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = {int(p.name[0]): p for p in (APP / "pages").glob("*.py")}
BANNER = "Розрахунки на завантаженому файлі"


def _user_df(n: int = 200, **override: object) -> pd.DataFrame:
    df, errors = validate_user_file(user_csv(n, **override), "x.csv")
    assert df is not None, errors
    return df


def _run(page: int, user_df: pd.DataFrame | None = None, **state: object) -> AppTest:
    at = AppTest.from_file(str(PAGES[page]), default_timeout=240)
    if user_df is not None:
        at.session_state["user_df"] = user_df
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _texts(at: AppTest) -> str:
    elements = [*at.markdown, *at.caption, *at.info, *at.warning, *at.error, *at.success]
    return " ".join(e.value for e in elements)


@pytest.mark.parametrize("page", range(1, 9))
def test_banner_on_every_page_with_user_data(page: int) -> None:
    at = _run(page, _user_df(1200))
    assert not at.exception
    assert any(BANNER in w.value and "1 200" in w.value for w in at.warning)
    assert any(b.key == "back_to_nhanes" for b in at.button)


@pytest.mark.parametrize("page", [4, 6, 7, 8])
def test_nhanes_only_pages_say_so_in_banner(page: int) -> None:
    at = _run(page, _user_df())
    assert any(BANNER in w.value and "NHANES" in w.value for w in at.warning)


def test_no_banner_without_user_data() -> None:
    assert not any(BANNER in w.value for w in _run(2).warning)


def test_back_button_on_page_1_resets_state_and_uploader() -> None:
    at = _run(1, _user_df(), user_file_id="abc", user_errors=[])
    assert at.session_state["user_df"] is not None
    at.button(key="back_to_nhanes").click().run()
    assert not at.exception
    assert "user_df" not in at.session_state and "user_file_id" not in at.session_state
    assert at.session_state["uploader_counter"] == 1
    assert not any(BANNER in w.value for w in at.warning)


def test_user_data_survives_page_reruns() -> None:
    at = _run(1, _user_df(), user_file_id="abc")
    at.run()
    at.run()
    assert not at.exception
    assert "user_df" in at.session_state
    assert any(BANNER in w.value for w in at.warning)


def test_removing_file_in_uploader_clears_user_data() -> None:
    at = _run(1)
    at.file_uploader[0].upload("x.csv", user_csv(200), "text/csv")
    at.run()
    assert "user_df" in at.session_state
    at.file_uploader[0].clear().run()
    assert not at.exception
    assert "user_df" not in at.session_state
    assert not any(BANNER in w.value for w in at.warning)


def test_page_1_new_tables_and_xlsx_template() -> None:
    at = _run(1)
    captions = [c.value for c in at.caption]
    assert "Таблиця 1.1. Кроки відбору учасників" in captions
    assert "Таблиця 1.4. Попередня обробка даних" in captions
    assert "Таблиця 1.5. Опис колонок шаблону для власного файлу" in captions
    keys = {b.key for b in at.get("download_button")}
    assert {"dl_template", "dl_template_xlsx"} <= keys
    assert "Browse files" not in _texts(at) and "аналітичної вибірки" in _texts(at)
    assert "Upload" not in _texts(at)  # кнопку завантажувача підписує блок стилів
    # підказку до завеликого файла перекладає блок стилів, тож підпис її не цитує
    assert "25 МБ" in _texts(at) and "File must be" not in _texts(at)
    assert "Файл має бути не більшим за 25 МБ" in at.html[0].value
    assert "когорт" not in _texts(at).lower()
    assert f"щонайменше {MIN_ROWS} рядків" in _texts(at)
    assert f"не менше ніж по {MIN_PER_CLASS}" in _texts(at)


def test_page_1_with_user_data_keeps_preprocessing_table() -> None:
    at = _run(1, _user_df())
    assert "Таблиця 1.4. Попередня обробка даних" in [c.value for c in at.caption]
    assert "Таблиця 1.1. Кроки відбору учасників" not in [c.value for c in at.caption]


def test_page_2_correlation_tables_are_collapsed_and_numbered() -> None:
    at = _run(2)
    labels = [e.label for e in at.expander]
    assert "Таблиця 2.2. Матриця кореляцій Пірсона" in labels
    assert "Таблиця 2.3. Матриця кореляцій Спірмена" in labels
    assert "Таблиця 2.4. Поширеність ЦД2: загалом і за циклами (незважена та зважена)" in [
        c.value for c in at.caption
    ]


def test_page_3_hypotheses_table_is_open() -> None:
    at = _run(3)
    assert not any("гіпотез" in e.label.lower() for e in at.expander)
    assert "Таблиця 3.1. Нульова та альтернативна гіпотези для кожного фактора" in [
        c.value for c in at.caption
    ]


def test_page_4_formula_unit_texts_and_thresholds_table() -> None:
    at = _run(4, _user_df())
    assert not at.exception
    latex = at.get("latex")[0].value
    assert "aligned" in latex and "\\text{вік}" in latex and "pa_h_week" not in latex
    assert "Таблиця 4.4. Пороги Юдена та відповідні чутливість і специфічність" in [
        e.label for e in at.expander
    ]
    text = _texts(at)
    assert "за середнім AUC крос-валідації на навчальній частині NHANES" in text
    assert "ту саму форму" in text and "тестовій вибірці NHANES" not in text.replace(
        "тестова вибірка використана лише для оцінки", ""
    )


def test_page_5_category_wording_and_home_texts() -> None:
    assert "чотири досяжні категорії з п'яти" in _texts(_run(5))
    home = AppTest.from_file(str(APP / "Головна.py"), default_timeout=60).run()
    text = _texts(home)
    assert "оцінками IDF та даними МОЗ України" in text and "офіційною статистикою" not in text
    assert any("Аналітична вибірка" in i.value for i in home.info)


def test_streamlit_toolbar_is_minimal() -> None:
    config = tomllib.loads((APP.parent / ".streamlit" / "config.toml").read_text("utf-8"))
    assert config["client"]["toolbarMode"] == "minimal"


# ---------- вироджені, але прийняті файли ----------
def _constant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("age", "male", "bmi", "waist", "pa_h_week", "pa_min_week"):
        out[col] = 1.0
    return out


@pytest.mark.parametrize("page", [2, 3])
def test_constant_columns_show_info_instead_of_crash(page: int) -> None:
    at = _run(page, _constant(_user_df()))
    assert not at.exception
    assert any("немає ознак з різними значеннями" in i.value for i in at.info)
    assert DISCLAIMER in _texts(at)


def test_page_2_with_single_varying_column() -> None:
    df = _constant(_user_df())
    df["age"] = np.arange(len(df)) % 50 + 20.0
    at = _run(2, df)
    assert not at.exception


def test_page_3_tiny_set_hides_unreliable_model_and_nan() -> None:
    tiny = _user_df(10, t2d=[0, 1] * 5)
    at = _run(3, tiny)
    assert not at.exception
    assert any("замало повних випадків" in i.value for i in at.info)
    for d in at.dataframe:
        values = d.value.astype(str).to_numpy().ravel()
        assert not {"nan", "None", "NaN"} & set(values)
    assert not any("багатофакторних" in c.value for c in at.caption)


def test_page_3_hides_forest_when_model_did_not_converge() -> None:
    df = _user_df(200)
    df["t2d"] = (df["age"] > 50).astype(float)
    at = _run(3, df)
    assert not at.exception
    forest = [c.value for c in at.caption if "багатофакторних відношень шансів" in c.value]
    assert any("не збіглася" in w.value for w in at.warning)
    assert forest == []
