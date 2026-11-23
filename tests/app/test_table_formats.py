"""Формат таблиць із перенесенням рядків: однакові знаки в колонці, пробіл тисяч."""

import sys
from pathlib import Path

import pandas as pd

# `from _common import …` потребує app/ у sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from _common import _dataframe_data, _decimals, _float_config, _wrap_styler

NBSP = " "


def test_decimals_is_max_needed() -> None:
    assert _decimals(pd.Series([0.502, 0.52, 0.512])) == 3
    assert _decimals(pd.Series([6.0, 7.0, None])) == 0
    assert _decimals(pd.Series([], dtype=float)) == 0


def test_wrap_styler_formats_columns() -> None:
    df = pd.DataFrame(
        {"F1": [0.502, 0.52], "N": [11304, 5], "k": [66.0, 378.0], "p": ["< 0.001", None]}
    )
    text = _wrap_styler(df).to_html()
    assert "0.520" in text
    assert "11 304" in text
    assert ">378<" in text
    assert "< 0.001" in text
    assert "—" in text


def test_float_config_uses_column_decimals() -> None:
    cfg = _float_config(pd.DataFrame({"a": [0.5, 0.25], "b": [1, 2]}))
    assert list(cfg) == ["a"]


def test_dataframe_counts_have_thousands_space_like_static_tables() -> None:
    df = pd.DataFrame(
        {
            "N": [10661, 5],
            "З ЦД2": [1234.0, float("nan")],  # кількість із пропуском — float
            "AUC": [0.777, 0.5],
            "SEQN": [93703, 93704],
            "Рік": [2017, 2020],
        }
    )
    shown = _dataframe_data(df)
    assert not isinstance(shown, pd.DataFrame)
    html = shown.to_html()
    assert f"10{NBSP}661" in html and f"1{NBSP}234" in html
    assert ">93703<" in html and ">2017<" in html  # коди й роки без пробілу тисяч
    assert shown.data is df  # значення лишаються числами: сортування за числом
    static = _wrap_styler(df).to_html()
    assert f"10{NBSP}661" in static and f"1{NBSP}234" in static and ">93703<" in static
    assert list(_float_config(df)) == ["AUC"]  # формат колонки перекрив би пробіл тисяч


def test_dataframe_without_counts_is_unchanged() -> None:
    df = pd.DataFrame({"Фактор": ["Вік"], "OR": [1.5]})
    assert _dataframe_data(df) is df
