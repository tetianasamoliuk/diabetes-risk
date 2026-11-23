"""Експорт результатів у Excel (форматування, вбудовані PNG) та ZIP із графіками."""

import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

import pandas as pd
from openpyxl.cell.cell import Cell
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from risk.config import DISCLAIMER, TOPIC
from risk.interpret import fmt_p

WORKBOOK_TITLE = TOPIC
MAX_SHEET_LEN = 31
MAX_COL_WIDTH = 60
IMG_WIDTH = 640
ROW_PX = 20  # висота рядка за замовчуванням, px
FIG_SHEET = "Графіки"
GLOSSARY_SHEET = "Пояснення"
_BAD_SHEET = re.compile(r"[\[\]:*?/\\]")
_BAD_FILE = re.compile(r'[\[\]:*?/\\<>|"]')
# «p» як окремий токен: на початку/після роздільника і перед кінцем/роздільником
_P_COL = re.compile(r"(^|[\s_(/])p($|[\s_(),-])", re.IGNORECASE)
# колонки-лічильники: цілі числа навіть якщо dtype float (через NaN)
_COUNT_COL = re.compile(r"^(n|N|N .*|З ЦД2|Позитивних|Кількість.*)$")
# відсотки: «…, %» або «% …» (а «95% CI від» для OR — ні)
_PCT_COL = re.compile(r",\s*%$|^%\s")
# три знаки: коефіцієнти, AUC і різниці AUC, метрики-частки, калібрування, розмір ефекту
_PRECISE_COL = re.compile(
    r"Коефіцієнт|Різниця|AUC|Поріг|Чутливість|Специфічність|Частка правильних|Точність|F1"
    r"|Браєра|ефект|Нахил|Зсув|^r\b"
)
COUNT_FORMAT = "#,##0"  # роздільник тисяч — за локаллю Excel (в українській — пробіл)
_HEADER_FILL = PatternFill("solid", start_color="D9E1F2")


@dataclass(frozen=True, slots=True)
class ExcelSheet:
    """Один лист: назва, таблиця, примітка над нею і формат дробових чисел (для кореляцій)."""

    name: str
    frame: pd.DataFrame
    note: str | None = None
    float_format: str | None = None


def is_p_column(name: object) -> bool:
    """Чи є колонка p-значенням (токен «p» у назві, а не просто літера «p»)."""
    return bool(_P_COL.search(str(name)))


def _safe_sheet_name(name: str) -> str:
    return _BAD_SHEET.sub("-", name)[:MAX_SHEET_LEN]


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    """Індекс -> колонка (якщо не RangeIndex), MultiIndex-колонки -> «a / b», p -> текст."""
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [" / ".join(map(str, c)) for c in frame.columns]
    if frame.index.names != [None] or not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    for name in frame.columns:
        if is_p_column(name) and pd.api.types.is_numeric_dtype(frame[name]):
            frame[name] = [fmt_p(v) if pd.notna(v) else "" for v in frame[name]]
    return frame


def _fit_width(cells: Iterable[Cell]) -> int:
    """Ширина за найдовшим текстом комірок + 2 на поля, не більше MAX_COL_WIDTH."""
    longest = max((len(str(c.value)) for c in cells if c.value is not None), default=0)
    return min(longest + 2, MAX_COL_WIDTH)


def autosize_columns(ws: Worksheet) -> None:
    """Ширина колонки = найдовший текст у ній (≤ 60)."""
    for col in ws.columns:
        ws.column_dimensions[get_column_letter(col[0].column)].width = _fit_width(col)


def number_format(name: str, dtype: object, float_format: str | None = None) -> str | None:
    """Формат чисел колонки за її назвою; для нечислових і bool — None.

    Лічильники — цілі з роздільником тисяч, відсотки — «0.0», коефіцієнти й AUC — «0.000»,
    решта — «0.00».
    """
    is_number = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    if not is_number:
        return None
    if _COUNT_COL.match(name):
        return COUNT_FORMAT
    if pd.api.types.is_integer_dtype(dtype):
        return "0"
    if _PCT_COL.search(name):
        return "0.0"
    return float_format or ("0.000" if _PRECISE_COL.search(name) else "0.00")


def apply_number_formats(
    ws: Worksheet, frame: pd.DataFrame, header_row: int, float_format: str | None = None
) -> None:
    """Формати чисел за назвами колонок; p після _flatten вже текст (fmt_p)."""
    for j, (name, dtype) in enumerate(frame.dtypes.items(), start=1):
        if (fmt := number_format(str(name), dtype, float_format)) is None:
            continue
        for i in range(header_row + 1, header_row + 1 + len(frame)):
            ws.cell(row=i, column=j).number_format = fmt


def _style_header(ws: Worksheet, row: int, ncols: int) -> None:
    for j in range(1, ncols + 1):
        cell = ws.cell(row=row, column=j)
        cell.font = Font(bold=True)
        cell.fill = _HEADER_FILL


def _write_sheet(xw: pd.ExcelWriter, sheet: ExcelSheet) -> None:
    frame = _flatten(sheet.frame)
    name = _safe_sheet_name(sheet.name)
    offset = 1 if sheet.note else 0
    frame.to_excel(xw, sheet_name=name, index=False, startrow=offset)
    ws = xw.sheets[name]
    header_row = offset + 1
    if sheet.note:
        ws.cell(row=1, column=1, value=sheet.note).font = Font(italic=True)
    _style_header(ws, header_row, len(frame.columns))
    apply_number_formats(ws, frame, header_row, sheet.float_format)
    autosize_columns(ws)
    if sheet.note:  # довга примітка не має розтягувати першу колонку: ширина без рядка 1
        ws.column_dimensions["A"].width = max(
            _fit_width(ws["A"][1:]), len(str(frame.columns[0])) + 2
        )
    ws.freeze_panes = f"A{header_row + 1}"


def _write_figures(xw: pd.ExcelWriter, figures: dict[str, bytes]) -> None:
    ws = xw.book.create_sheet(FIG_SHEET)
    row = 1
    for caption, data in figures.items():
        ws.cell(row=row, column=1, value=caption).font = Font(bold=True)
        img = Image(io.BytesIO(data))
        img.height = round(img.height * IMG_WIDTH / img.width)
        img.width = IMG_WIDTH
        ws.add_image(img, f"A{row + 1}")
        row += ceil(img.height / ROW_PX) + 3  # підпис + картинка + відступ
    ws.column_dimensions["A"].width = IMG_WIDTH / 7


def build_workbook(
    sheets: list[ExcelSheet], figures: dict[str, bytes], glossary: pd.DataFrame
) -> bytes:
    """Збирає книгу: листи за порядком, потім «Графіки» і «Пояснення»."""
    gl = glossary.iloc[:, :2].set_axis(["Термін", "Пояснення"], axis=1)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet in sheets:
            _write_sheet(xw, sheet)
        _write_figures(xw, figures)
        _write_sheet(xw, ExcelSheet(GLOSSARY_SHEET, gl, note=DISCLAIMER))
        props = xw.book.properties
        props.creator, props.title = "", WORKBOOK_TITLE
    return buf.getvalue()


def _file_name(caption: str) -> str:
    return _BAD_FILE.sub("-", caption.strip()).replace(" ", "_") + ".png"


def figures_zip(figures: dict[str, bytes]) -> bytes:
    """ZIP з PNG; імена файлів походять із підписів."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for caption, data in figures.items():
            z.writestr(_file_name(caption), data)
    return buf.getvalue()
