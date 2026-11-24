"""Кеш рисунків: повторний прогін не малює заново, ключ — увесь вміст аргументів будівника."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
import streamlit as st
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from risk.data.user_data import validate_user_file
from risk.viz import figures
from risk.viz.style import fig_to_png
from tests.app.conftest import user_csv

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = {int(p.name[0]): p for p in (APP / "pages").glob("*.py")}
sys.path.insert(0, str(APP))
import _common

# скільки рисунків на сторінці з даними NHANES (сторінка 7 — після «Оцінити ризик»)
FIGURES_ON_PAGE = {1: 2, 2: 5, 3: 2, 4: 7, 5: 2, 6: 1, 7: 1}


def _run(page: int, user_df: pd.DataFrame | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGES[page]), default_timeout=240)
    if user_df is not None:
        at.session_state["user_df"] = user_df
    return at.run()


def _hist(at: AppTest) -> tuple[str, str]:
    """Рисунок 2.1: URL зображення й файла кнопки (Streamlit будує URL з хешу вмісту)."""
    return at.image[0].value[0], at.download_button(key="dl_figure_2.1").proto.url


def _counted(calls: dict[str, int], name: str, func: Callable[..., object]) -> Callable:
    def wrapper(*args: object, **kwargs: object) -> object:
        calls[name] += 1
        return func(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize("page", [2, 4])
def test_rerun_does_not_build_or_render_figures_again(
    page: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"build": 0, "png": 0}
    monkeypatch.setattr(_common, "fig_to_png", _counted(calls, "png", _common.fig_to_png))
    monkeypatch.setattr(figures, "_new", _counted(calls, "build", figures._new))
    monkeypatch.setattr(figures, "_shap_figure", _counted(calls, "build", figures._shap_figure))
    st.cache_data.clear()
    at = _run(page)
    first = dict(calls)
    at.run()
    assert not at.exception
    assert first["build"] > 0 and first["png"] > 0
    assert calls == first


def test_histogram_follows_selected_feature() -> None:
    at = _run(2)
    assert at.selectbox[0].value == "age"
    age = _hist(at)
    at.selectbox[0].set_value("bmi").run()
    bmi = _hist(at)
    assert bmi[0] != age[0] and bmi[1] != age[1]
    at.selectbox[0].set_value("age").run()
    assert _hist(at) == age


def test_user_file_gets_its_own_figure() -> None:
    user_df, errors = validate_user_file(user_csv(1200), "x.csv")
    assert user_df is not None, errors
    nhanes, user = _hist(_run(2)), _hist(_run(2, user_df))
    assert user[0] != nhanes[0] and user[1] != nhanes[1]


@pytest.mark.parametrize("page", FIGURES_ON_PAGE)
def test_page_keeps_figure_buttons_and_images(page: int) -> None:
    at = _run(page)
    if page == 7:
        at.button[0].click().run()
    keys = [b.key for b in at.download_button if str(b.key).startswith("dl_figure_")]
    assert keys == [f"dl_figure_{page}.{k}" for k in range(1, FIGURES_ON_PAGE[page] + 1)]
    assert len(at.image) == FIGURES_ON_PAGE[page]


def test_pngs_are_the_figure_at_200_and_300_dpi() -> None:
    _common._figure_pngs.clear()
    args = (validate_user_file(user_csv(300), "x.csv")[0], "bmi", "ІМТ")
    build = figures.hist_by_group
    screen, download = _common._figure_pngs("hist", build, args, {})
    assert screen == fig_to_png(build(*args), 200)
    assert download == fig_to_png(build(*args), 300)


def _builds(calls: list[tuple[str, tuple]]) -> list[bool]:
    """Для кожного виклику кешу (ім'я, аргументи) — чи будувався рисунок заново."""
    _common._figure_pngs.clear()
    built: list[object] = []

    def build(*args: object) -> Figure:
        built.append(args)
        return Figure(figsize=(1, 1))

    result = []
    for name, args in calls:
        before = len(built)
        _common._figure_pngs(name, build, args, {})
        result.append(len(built) > before)
    return result


def _expl(
    values: tuple[float, ...] = (0.1, -0.2),
    base: float = 0.5,
    data: tuple[float, ...] = (1.0, 26.0),
    names: tuple[str, ...] = ("bmi", "age"),
) -> shap.Explanation:
    """Пояснення з даними-текстами, як у readable_explanation: щоразу нові об'єкти рядків."""
    shown = np.array([f"{v:.1f}" for v in data], dtype=object)
    return shap.Explanation(
        values=np.array(values), base_values=base, data=shown, feature_names=list(names)
    )


def test_shap_explanation_key_is_its_whole_content() -> None:
    variants = [
        _expl(),
        _expl(),  # той самий вміст в інших об'єктах — з кешу
        _expl(values=(0.1, -0.3)),
        _expl(base=0.6),
        _expl(data=(0.0, 26.0)),
        _expl(names=("bmi", "waist")),
    ]
    assert _builds([("shap", (e,)) for e in variants]) == [True, False, True, True, True, True]


def test_builder_name_is_part_of_key() -> None:
    args = (pd.DataFrame({"x": [1.0, 2.0]}),)
    assert _builds([("a", args), ("a", args), ("b", args)]) == [True, False, True]


@pytest.mark.parametrize("series", [False, True])
def test_big_table_key_covers_every_row(series: bool) -> None:
    df = pd.DataFrame({"x": np.arange(60_000, dtype=float)})
    # рядок поза вибіркою 10 000 рядків, яку Streamlit хешує для таблиць від 50 000 рядків
    row = df.index.difference(df.sample(n=10_000, random_state=0).index)[0]
    other = df.copy()
    other.loc[row, "x"] = -1.0
    a, b = (df["x"], other["x"]) if series else (df, other)
    assert _builds([("big", (a,)), ("big", (b,))]) == [True, True]
