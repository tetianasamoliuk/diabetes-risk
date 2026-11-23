"""Валідація користувацького CSV/Excel: шаблон, перевірки, нормалізація."""

import csv
import io
import re
import warnings
from collections import Counter
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from risk.config import FEATURE_LABELS_UK
from risk.interpret import plural_uk

BINARY = ["male", "fam_dm", "bp_med", "t2d"]
REQUIRED_COLUMNS = ["age", "male", "bmi", "waist", "t2d"]
MIN_ROWS = 10
MAX_ERRORS = 20
MIN_PER_CLASS = 5
CASES_PER_PARAMETER = 10
SHOWN_ROWS = 3

TABLE_HINT = "Збережіть файл як CSV UTF-8 або Excel (.xlsx) і спробуйте знову"
NOT_UTF8_MSG = (
    "Не вдалося прочитати файл: це не текстовий CSV у кодуванні UTF-8. "
    "Збережіть як CSV UTF-8 або Excel (.xlsx)"
)
XLSX_FAILED_MSG = "Не вдалося відкрити файл як Excel (.xlsx). Збережіть файл заново у форматі .xlsx"
READ_FAILED_MSG = f"Не вдалося прочитати файл як таблицю. {TABLE_HINT}"
EXTRA_FIELDS_MSG = (
    "У рядках більше значень, ніж назв колонок у заголовку (зайва кома в кінці рядків?)"
)
HEADERS_ONLY_MSG = "У файлі є лише заголовки, немає рядків даних"
LOW_VARIETY_MSG = (
    "Замало варіативних даних для аналізу: усі числові колонки мають одне значення "
    "або жодного рядка немає заповненим по всіх колонках"
)

HEADERS_UNKNOWN_MSG = (
    "Жодної назви колонки не розпізнано. Назви мають бути точно такими, як у шаблоні "
    "(англійськими літерами, наприклад age, bmi); шаблон можна завантажити на цій сторінці"
)

RANGES: dict[str, tuple[float, float]] = {
    "age": (20, 120),
    "bmi": (10, 100),
    "waist": (40, 200),
    "sbp": (60, 260),
    "dbp": (30, 160),
    "tc": (50, 600),
    "hdl": (5, 250),
    "pa_h_week": (0, 168),
}

# Описи колонок: назва з FEATURE_LABELS_UK + допустимі значення для бінарних
TEMPLATE_COLUMNS: dict[str, str] = {
    c: FEATURE_LABELS_UK[c] + (": 0 або 1" if c in BINARY else "")
    for c in [
        "age",
        "male",
        "bmi",
        "waist",
        "sbp",
        "dbp",
        "tc",
        "hdl",
        "pa_h_week",
        "fam_dm",
        "bp_med",
        "t2d",
    ]
}

_EXAMPLES = [
    [45, 1, 27.5, 96, 128, 82, 190, 48, 3.5, 0, 0, 0],
    [58, 0, 33.1, 104, 145, 90, 225, 41, 1.0, 1, 1, 1],
    [31, 0, 22.4, 74, 112, 70, 170, 62, 5.0, 0, 0, 0],
]


def template_csv() -> bytes:
    """CSV-шаблон (UTF-8 з BOM для Excel): заголовок і 3 приклади."""
    df = pd.DataFrame(_EXAMPLES, columns=list(TEMPLATE_COLUMNS))
    return df.to_csv(index=False).encode("utf-8-sig")


def template_xlsx() -> bytes:
    """Excel-шаблон: аркуш «Дані» (заголовок і 3 приклади) та аркуш «Опис колонок»."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(_EXAMPLES, columns=list(TEMPLATE_COLUMNS)).to_excel(
            writer, sheet_name="Дані", index=False
        )
        template_description().to_excel(writer, sheet_name="Опис колонок", index=False)
    return buf.getvalue()


def template_description() -> pd.DataFrame:
    """Таблиця для сторінки: колонка, опис, діапазон, обов'язкова."""
    rows = []
    for col, desc in TEMPLATE_COLUMNS.items():
        lo_hi = RANGES.get(col)
        rng = f"{_human(lo_hi[0])}–{_human(lo_hi[1])}" if lo_hi else "0 або 1"
        rows.append((col, desc, rng, "так" if col in REQUIRED_COLUMNS else "ні"))
    return pd.DataFrame(rows, columns=["Колонка", "Опис", "Діапазон", "Обов'язкова"])


_DECIMAL_COMMA = re.compile(r"-?\d+(,\d+)?")
_MANGLED = re.compile(r"(.+)\.\d+")  # pandas перейменовує дублікати: age, age.1
_READ_ERRORS = (
    ValueError,
    OSError,
    csv.Error,
    pd.errors.ParserError,
    KeyError,
    TypeError,
    IndexError,
)


def _human(x: float) -> str:
    """Число без експоненти: 1000000000 → «1 000 000 000», inf → «∞»."""
    if not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:,.4f}".rstrip("0").rstrip(".").replace(",", " ").replace("-", "−")


def _fix_decimal_comma(df: pd.DataFrame) -> pd.DataFrame:
    """Текстові колонки з «27,5» (українські Excel/CSV з «;») → числа; потрібна хоч одна кома."""
    for col in df.select_dtypes(exclude="number").columns:
        s = df[col].dropna().astype(str).str.strip()
        if s.str.contains(",").any() and s.str.fullmatch(_DECIMAL_COMMA).all():
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", "."))
    return df


def _duplicate_error(cols: list[str]) -> str | None:
    """Повідомлення про першу колонку, що трапляється двічі (з урахуванням «age.1»)."""
    for name, n in Counter(cols).items():
        if n > 1:
            return f'Колонка "{name}" трапляється двічі'
    for c in cols:
        if (m := _MANGLED.fullmatch(c)) and m[1] in cols:
            return f'Колонка "{m[1]}" трапляється двічі'
    return None


def _read_table(buf: io.BytesIO, name: str) -> pd.DataFrame | str:
    """DataFrame або текст помилки для невідомого розширення / пошкодженого .xlsx."""
    if name.endswith(".csv"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                buf, sep=None, engine="python", encoding="utf-8-sig", index_col=False
            )
    if not name.endswith(".xlsx"):
        return "Підтримуються лише .csv і .xlsx"
    try:
        return pd.read_excel(buf, index_col=None)
    except (ValueError, OSError, BadZipFile, KeyError):
        return XLSX_FAILED_MSG


def _read(content: bytes, filename: str) -> pd.DataFrame | str:
    """Таблиця з нормалізованими назвами колонок або текст помилки читання."""
    if not content.strip():
        return "Файл порожній"
    try:
        df = _read_table(io.BytesIO(content), filename.lower())
    except pd.errors.EmptyDataError:
        return "Файл порожній"
    except UnicodeDecodeError:
        return NOT_UTF8_MSG
    except pd.errors.ParserWarning:
        return EXTRA_FIELDS_MSG
    except _READ_ERRORS:
        return READ_FAILED_MSG
    if isinstance(df, str):
        return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    if err := _duplicate_error(list(df.columns)):
        return err
    return _fix_decimal_comma(df)


def _label(col: str) -> str:
    return FEATURE_LABELS_UK[col].split(",")[0].split(" (")[0]


def _row_numbers(mask: pd.Series | np.ndarray) -> list[int]:
    """Номери рядків файла (заголовок = рядок 1) за позиціями True."""
    return [int(i) + 2 for i in np.flatnonzero(np.asarray(mask))]


def _rows_text(rows: list[int]) -> str:
    """«рядку 5», «рядках 2, 3», «рядках 2, 3, 4 … (ще 7)»."""
    shown = ", ".join(map(str, rows[:SHOWN_ROWS]))
    if len(rows) > SHOWN_ROWS:
        shown += f" … (ще {len(rows) - SHOWN_ROWS})"
    return f"{'рядку' if len(rows) == 1 else 'рядках'} {shown}"


def _binary_errors(col: str, raw: pd.Series, num: pd.Series) -> list[str]:
    bad = _row_numbers(raw.notna() & ~num.isin([0, 1]))
    if not bad:
        return []
    return [
        f'У колонці "{col}" допустимі лише 0 або 1; інші значення в {_rows_text(bad)}. '
        "Відповіді «так/ні» запишіть як 1/0"
    ]


def _range_errors(col: str, num: pd.Series) -> list[str]:
    lo, hi = RANGES[col]
    values = num.to_numpy(dtype=float)
    errs = []
    for r in _row_numbers(~np.isnan(values) & ((values < lo) | (values > hi))):
        errs.append(
            f"{_label(col)} {_human(values[r - 2])} поза допустимим діапазоном "
            f"{_human(lo)}–{_human(hi)} (рядок {r})"
        )
    return errs


def _missing_errors(col: str, raw: pd.Series) -> list[str]:
    """Одне повідомлення, якщо колонка порожня повністю; інакше по рядках."""
    if raw.isna().all():
        return [f'Колонка "{col}" повністю порожня']
    return [f'У колонці "{col}" є пропуск у рядку {r}' for r in _row_numbers(raw.isna())]


def _check_column(col: str, raw: pd.Series) -> tuple[pd.Series, list[str]]:
    """Перевіряє одну колонку; повертає числову серію і помилки."""
    num = pd.to_numeric(raw, errors="coerce")
    if col in BINARY:
        errs = _binary_errors(col, raw, num)
    else:
        text = _row_numbers(raw.notna() & num.isna())
        errs = [f'У колонці "{col}" є текст у рядку {r}' for r in text]
        errs += _range_errors(col, num)
    if col in REQUIRED_COLUMNS:
        errs += _missing_errors(col, raw)
    return num, errs


def _consistency_errors(data: pd.DataFrame) -> list[str]:
    """Перевірки між колонками одного рядка: САТ > ДАТ, загальний ХС ≥ ЛПВЩ."""
    rules = (
        ("sbp", "dbp", lambda a, b: a <= b, "Систолічний тиск має бути вищим за діастолічний"),
        ("tc", "hdl", lambda a, b: a < b, "Загальний холестерин не може бути меншим за ЛПВЩ"),
    )
    errs = []
    for a, b, broken, text in rules:
        for r in _row_numbers(broken(data[a], data[b]).to_numpy()):
            errs.append(f"{text} (рядок {r})")
    return errs


def _low_variety(data: pd.DataFrame) -> bool:
    """Немає змінних із різними значеннями або жодного рядка без пропусків по них."""
    varying = [c for c in RANGES if data[c].nunique() > 1]
    return not varying or data[[*varying, "t2d"]].dropna().empty


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Додає службові колонки, що очікують FINDRISC і статистика."""
    out = df.copy()
    out["pa_min_week"] = out["pa_h_week"] * 60
    out["cycle_2017"] = 0.0
    for col in ("diq010", "high_glu_hist", "t2d_undiag"):
        out[col] = np.nan
    return out


def _limit(errors: list[str]) -> list[str]:
    """Перші MAX_ERRORS помилок і підсумок про решту."""
    if len(errors) <= MAX_ERRORS:
        return errors
    rest = len(errors) - MAX_ERRORS
    word = plural_uk(rest, ("помилка", "помилки", "помилок"))
    return errors[:MAX_ERRORS] + [f"… та ще {rest} {word}"]


def _columns_errors(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors = [f'Немає обов\'язкової колонки "{c}"' for c in REQUIRED_COLUMNS if c not in raw]
    if not any(c in raw for c in TEMPLATE_COLUMNS):
        errors.append(HEADERS_UNKNOWN_MSG)
    data = pd.DataFrame(index=range(len(raw)))
    for col in TEMPLATE_COLUMNS:
        if col in raw:
            data[col], errs = _check_column(col, raw[col].reset_index(drop=True))
            errors += errs
        else:
            data[col] = np.nan
    return data, errors


def _dataset_errors(data: pd.DataFrame) -> list[str]:
    """Помилки всього набору, коли окремі значення вже коректні."""
    classes = data["t2d"].value_counts().reindex([0, 1], fill_value=0)
    if not (classes >= MIN_PER_CLASS).all():
        return [f'У колонці "t2d" мають бути і 0, і 1 (щонайменше по {MIN_PER_CLASS} спостережень)']
    return [LOW_VARIETY_MSG] if _low_variety(data) else []


def validate_user_file(content: bytes, filename: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Читає і перевіряє файл. df = None, якщо є хоч одна помилка."""
    raw = _read(content, filename)
    if isinstance(raw, str):
        return None, [raw]
    if raw.empty:
        return None, [HEADERS_ONLY_MSG if len(raw.columns) else "Файл порожній"]
    data, errors = _columns_errors(raw)
    if len(raw) < MIN_ROWS:
        errors.append(f"Замало рядків для аналізу: {len(raw)}, потрібно щонайменше {MIN_ROWS}")
    errors += _consistency_errors(data)
    if not errors:
        errors = _dataset_errors(data)
    if errors:
        return None, _limit(errors)
    return _finalize(data.astype(float)), []


def complete_cases_shortfall(df: pd.DataFrame, features: list[str]) -> str | None:
    """Пояснення, якщо повних випадків замало для багатофакторної моделі; інакше None."""
    need = CASES_PER_PARAMETER * (len(features) + 1)
    have = len(df.dropna(subset=[*features, "t2d"]))
    if have >= need:
        return None
    return (
        f"Для багатофакторної моделі замало повних випадків: потрібно щонайменше {need}, "
        f"є {have}. Заповніть більше рядків або залиште менше необов'язкових колонок."
    )
