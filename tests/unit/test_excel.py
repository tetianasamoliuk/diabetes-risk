"""Тести експорту в Excel."""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from risk.config import DISCLAIMER, TOPIC
from risk.report.excel import (
    ExcelSheet,
    apply_number_formats,
    build_workbook,
    figures_zip,
    is_p_column,
    number_format,
)
from risk.viz.style import close

_GL = pd.DataFrame({"a": ["AUC"], "b": ["x"]})
LONG = "Дуже довга назва листа з багатьма символами понад ліміт"


def _png(seed: int) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.random.default_rng(seed).random(10))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    close(fig)
    return buf.getvalue()


@pytest.fixture(scope="module")
def figures() -> dict[str, bytes]:
    return {"Рисунок 2.1. ROC-криві": _png(1), "Рисунок 2.2. Калібровка/тест": _png(2)}


@pytest.fixture(scope="module")
def wb_bytes(figures: dict[str, bytes]) -> bytes:
    plain = pd.DataFrame(
        {"var": ["a", "b"], "OR": [1.234, 2.5], "n": [10, 20], "p": [0.01234, 0.5]}
    )
    named = pd.DataFrame({"p_holm": [0.04], "stat": [3.14159]}, index=pd.Index(["x"], name="Тест"))
    multi = pd.DataFrame(
        [[1.0, 2.0]],
        columns=pd.MultiIndex.from_tuples([("AUC", "lo"), ("AUC", "hi")]),
    )
    sheets = [
        ExcelSheet("Опис", plain),
        ExcelSheet("Гіпотези", named, note="Примітка до листа"),
        ExcelSheet(f"{LONG}[:]", multi),
    ]
    gl = pd.DataFrame({"a": ["AUC"], "b": ["площа під кривою"]})
    return build_workbook(sheets, figures, gl)


@pytest.fixture(scope="module")
def wb(wb_bytes: bytes) -> Workbook:
    return load_workbook(io.BytesIO(wb_bytes))


def test_sheet_order_and_names(wb: Workbook) -> None:
    names = wb.sheetnames
    assert names[:2] == ["Опис", "Гіпотези"]
    assert names[-2:] == ["Графіки", "Пояснення"]
    assert len(names[2]) == 31
    assert not set(names[2]) & set("[]:*?/\\")


def test_workbook_properties_are_neutral(wb: Workbook) -> None:
    assert not wb.properties.creator
    assert wb.properties.title == TOPIC


def test_header_bold_and_freeze(wb: Workbook) -> None:
    ws = wb["Опис"]
    assert ws["A1"].font.b
    assert ws.freeze_panes == "A2"
    ws = wb["Гіпотези"]
    assert ws["A1"].value == "Примітка до листа"
    assert ws["A1"].font.i
    assert ws["A2"].font.b
    assert ws.freeze_panes == "A3"


def test_number_formats(wb: Workbook) -> None:
    ws = wb["Опис"]
    assert ws["B2"].number_format == "0.00"
    assert ws["C2"].number_format == "#,##0"
    assert ws["D2"].value == "0.012"  # p — текст, як у таблицях 3.2/4.1


def test_named_index_and_p_holm(wb: Workbook) -> None:
    ws = wb["Гіпотези"]
    assert [c.value for c in ws[2]] == ["Тест", "p_holm", "stat"]
    assert ws["B3"].value == "0.040"
    assert ws["C3"].number_format == "0.00"


def test_multiindex_flattened(wb: Workbook) -> None:
    ws = wb.worksheets[2]
    assert [c.value for c in ws[1]] == ["AUC / lo", "AUC / hi"]


def test_autosize(wb: Workbook) -> None:
    ws = wb["Гіпотези"]
    assert ws.column_dimensions["A"].width >= len("Тест")
    assert all(d.width <= 62 for d in ws.column_dimensions.values())


def test_figures_sheet(wb: Workbook) -> None:
    ws = wb["Графіки"]
    assert len(ws._images) == 2
    assert all(i.anchor.ext.width == 640 * 9525 for i in ws._images)
    assert ws["A1"].value == "Рисунок 2.1. ROC-криві"
    assert ws._images[0].anchor._from.row == 1


def test_glossary(wb: Workbook) -> None:
    ws = wb["Пояснення"]
    assert ws["A1"].value == DISCLAIMER
    assert [c.value for c in ws[2]] == ["Термін", "Пояснення"]
    assert ws["A3"].value == "AUC"


def test_tiny_p_is_text_and_counts_are_int() -> None:
    frame = pd.DataFrame({"p": [1.7e-220], "N": [10.0], "З ЦД2": [3.0], "x": [1.5]})
    book = load_workbook(io.BytesIO(build_workbook([ExcelSheet("t", frame)], {}, _GL)))
    ws = book["t"]
    assert ws["A2"].value == "< 0.001"
    assert ws["B2"].number_format == "#,##0"
    assert ws["C2"].number_format == "#,##0"
    assert ws["D2"].number_format == "0.00"


def test_range_index_not_added() -> None:
    frame = pd.DataFrame({"a": [1]}, index=pd.RangeIndex(5, 6))
    ws = load_workbook(io.BytesIO(build_workbook([ExcelSheet("t", frame)], {}, _GL)))["t"]
    assert [c.value for c in ws[1]] == ["a"]


def test_figures_zip(figures: dict[str, bytes]) -> None:
    with zipfile.ZipFile(io.BytesIO(figures_zip(figures))) as z:
        names = z.namelist()
        assert names == ["Рисунок_2.1._ROC-криві.png", "Рисунок_2.2._Калібровка-тест.png"]
        assert [z.read(n) for n in names] == list(figures.values())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p", True),
        ("p_holm", True),
        ("p (adj)", True),
        ("Вік / p", True),
        ("pa_h_week", False),
        ("OR", False),
        ("sbp", False),
    ],
)
def test_is_p_column(name: str, expected: bool) -> None:
    assert is_p_column(name) is expected


def test_apply_number_formats_skips_bool() -> None:
    ws = Workbook().active
    frame = pd.DataFrame({"flag": [True, False], "v": [1.5, 2.5]})
    apply_number_formats(ws, frame, header_row=1)
    assert ws.cell(row=2, column=1).number_format == "General"
    assert ws.cell(row=2, column=2).number_format == "0.00"


@pytest.mark.parametrize(
    ("name", "dtype", "expected"),
    [
        ("N", "float64", "#,##0"),
        ("Місце", "int64", "0"),
        ("Незважена частка, %", "float64", "0.0"),
        ("% з ЦД2", "float64", "0.0"),
        ("95% ДІ від", "float64", "0.00"),
        ("Коефіцієнт", "float64", "0.000"),
        ("Різниця AUC (A − B)", "float64", "0.000"),
        ("r (рангово-бісеріальна)", "float64", "0.000"),
        ("OR", "float64", "0.00"),
        ("Фактор", "object", None),
    ],
)
def test_number_format(name: str, dtype: str, expected: str | None) -> None:
    assert number_format(name, np.dtype(dtype)) == expected


def test_sheet_float_format_overrides_default() -> None:
    frame = pd.DataFrame({"Змінна": ["Вік"], "Вік": [1.0], "N": [5.0]})
    sheet = ExcelSheet("t", frame, float_format="0.000")
    ws = load_workbook(io.BytesIO(build_workbook([sheet], {}, _GL)))["t"]
    assert ws["B2"].number_format == "0.000"
    assert ws["C2"].number_format == "#,##0"
