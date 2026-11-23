"""Зовнішні дані: поширеність діабету в Україні (IDF, МОЗ) і довідкові оцінки NCHS для США.

Кожен рядок CSV має джерело, URL і дату перевірки; рядки без них відхиляються, щоб у таблицях
не з'являлися числа без походження.
"""

import re
from pathlib import Path

import pandas as pd

from risk.config import EXTERNAL_DIR
from risk.errors import DataValidationError
from risk.interpret import plural_uk

UKRAINE_CSV_COLUMNS = ["source", "year", "indicator", "value", "unit", "url", "accessed"]
US_CSV_COLUMNS = ["source", "period", "indicator", "value", "unit", "url", "accessed"]
SOURCE_COLUMNS = ["source", "url", "accessed"]
COMPARISON_COLUMNS = [
    "measure", "source", "period", "age", "method", "value", "ci_low", "ci_high", "accessed",
]  # fmt: skip
COMPARISON_COLS_UK = {
    "measure": "Показник", "source": "Джерело", "period": "Період", "age": "Вік",
    "method": "Метод", "value": "Поширеність, %", "ci_low": "95% ДІ від, %",
    "ci_high": "95% ДІ до, %", "accessed": "Дата перевірки",
}  # fmt: skip
OTHER_COLS_UK = {
    "source": "Джерело", "period": "Період", "indicator": "Показник", "value": "Значення",
    "unit": "Одиниця", "url": "URL", "accessed": "Дата перевірки",
}  # fmt: skip
# Що саме вимірює рядок таблиці порівняння; у цьому порядку рядки й ідуть
MEASURE_TOTAL, MEASURE_DIAG, MEASURE_UNDIAG = (
    "Діабет загалом", "Діагностований діабет", "Недіагностований діабет",
)  # fmt: skip
MEASURES = [MEASURE_TOTAL, MEASURE_DIAG, MEASURE_UNDIAG]
# Початок назви показника в us_reference.csv
US_TOTAL, US_DIAGNOSED, US_UNDIAGNOSED = "Усього", "Діагностований", "Недіагностований"
NCHS_METHOD = "глюкоза натще або HbA1c + самозвіт"
US_ROWS = (  # (початок назви показника NCHS, що вимірює, метод)
    (US_TOTAL, MEASURE_TOTAL, NCHS_METHOD),
    (US_DIAGNOSED, MEASURE_DIAG, "самозвіт про діагноз"),
    (US_UNDIAGNOSED, MEASURE_UNDIAG, "глюкоза натще або HbA1c без діагнозу"),
)
IDF_METHOD = "стандартизована за віком, модельна оцінка"
# Початок назви показника в ukraine_stats.csv для кількостей осіб
UA_COUNT_PREFIXES = {
    "crude": "Кількість дорослих 20–79 з діабетом",
    "undiag": "Кількість дорослих 20–79 з недіагностованим",
    "moz": "Пацієнтів із цукровим діабетом",
}
UA_POPULATION = "Доросле населення 20–79"
CALC_NOTE = "розрахунок із чисел джерела"
# Частки від дорослого населення 20–79 за IDF: ключ -> (що вимірює, як отримано)
DERIVED = {
    "crude": (MEASURE_TOTAL, "сира частка"),
    "diag": (MEASURE_DIAG, "усі з діабетом мінус недіагностовані"),
    "moz": (MEASURE_DIAG, "облік ЕСОЗ, усі типи діабету"),
    "undiag": (MEASURE_UNDIAG, "модельна оцінка IDF"),
}
MOZ_AGREE_TOL = 0.2  # облік МОЗ «узгоджується» з IDF, якщо відрізняється не більш ніж на 20 %
PATIENT_FORMS = ("пацієнт", "пацієнти", "пацієнтів")
_AGE = re.compile(r"\d+–\d+|≥ ?\d+")


def _has_text(col: pd.Series) -> pd.Series:
    return col.notna() & (col.astype(str).str.strip() != "")


def _read_sourced(path: Path, columns: list[str]) -> pd.DataFrame:
    """CSV з перевіркою колонок; рядки без джерела відкидаються, value має бути числом."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if list(df.columns) != columns:
        raise DataValidationError(f"Очікувані колонки {columns}, отримано {list(df.columns)}")
    df = df[df[SOURCE_COLUMNS].apply(_has_text).all(axis=1)].reset_index(drop=True)
    value = pd.to_numeric(df["value"], errors="coerce")
    if value.isna().any():  # +2: заголовок — рядок 1, нумерація з 1
        raise DataValidationError(f"рядок {value.isna().idxmax() + 2}: value не число")
    return df.assign(value=value.astype(float))


def load_ukraine_stats(path: Path = EXTERNAL_DIR / "ukraine_stats.csv") -> pd.DataFrame:
    """Дані по Україні; відсутній файл або лише заголовок → порожній DataFrame."""
    df = _read_sourced(path, UKRAINE_CSV_COLUMNS)
    return df.assign(year=df["year"].astype(int))


def load_us_reference(path: Path = EXTERNAL_DIR / "us_reference.csv") -> pd.DataFrame:
    """Довідкові оцінки NCHS для США; відсутній файл → порожній DataFrame."""
    return _read_sourced(path, US_CSV_COLUMNS)


def _main_index(ukraine: pd.DataFrame) -> int | None:
    """Найсвіжіша поширеність діабету в Україні (у %), з якою порівнюємо США."""
    prev = ukraine[(ukraine["unit"] == "%") & ukraine["indicator"].str.startswith("Поширеність")]
    return None if prev.empty else int(prev["year"].astype(int).idxmax())


def main_prevalence(ukraine: pd.DataFrame) -> pd.Series | None:
    """Рядок основної оцінки поширеності по Україні або None, якщо такої немає."""
    i = _main_index(ukraine)
    return None if i is None else ukraine.loc[i]


def ua_value(ukraine: pd.DataFrame, prefix: str) -> pd.Series | None:
    """Найсвіжіший рядок України, назва показника якого починається з prefix; None — немає."""
    found = ukraine[ukraine["indicator"].str.startswith(prefix)]
    return None if found.empty else found.loc[found["year"].idxmax()]


def ukraine_counts(ukraine: pd.DataFrame) -> dict[str, tuple[float, pd.Series]]:
    """Кількості осіб (crude — з діабетом, undiag, moz, diag = crude − undiag) і рядок-джерело."""
    found = {k: ua_value(ukraine, p) for k, p in UA_COUNT_PREFIXES.items()}
    out = {k: (float(r["value"]), r) for k, r in found.items() if r is not None}
    if {"crude", "undiag"} <= out.keys():
        out["diag"] = (out["crude"][0] - out["undiag"][0], out["crude"][1])
    return out


def ukraine_shares(ukraine: pd.DataFrame) -> dict[str, float]:
    """Частки (%) кількостей ukraine_counts від дорослого населення 20–79 за IDF.

    Сира частка (crude) — фактична поширеність, від неї залежить PPV; без населення — {}.
    """
    population = ua_value(ukraine, UA_POPULATION)
    if population is None:
        return {}
    counts = ukraine_counts(ukraine)
    return {k: float(100 * counts[k][0] / population["value"]) for k in DERIVED if k in counts}


def moz_vs_idf(ukraine: pd.DataFrame) -> float | None:
    """Пацієнти в ЕСОЗ (МОЗ) у % від оцінки IDF для діагностованих; None — бракує чисел."""
    counts = ukraine_counts(ukraine)
    if not {"moz", "diag"} <= counts.keys():
        return None
    return 100 * counts["moz"][0] / counts["diag"][0]


def _int(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def moz_idf_text(ukraine: pd.DataFrame) -> str | None:
    """Чи узгоджується облік МОЗ з оцінкою IDF для діагностованих (з чисел файлу)."""
    if (ratio := moz_vs_idf(ukraine)) is None:
        return None
    counts = ukraine_counts(ukraine)
    (idf, idf_row), (moz, moz_row) = counts["diag"], counts["moz"]
    head = (
        f"За IDF у {idf_row['year']} році в Україні було близько {_int(idf)} дорослих 20–79 "
        f"років із діагностованим діабетом (усі з діабетом мінус недіагностовані; {CALC_NOTE}). "
        f"В електронній системі охорони здоров'я у {moz_row['year']} році обліковано "
        f"{_int(moz)} {plural_uk(int(moz), PATIENT_FORMS)} з діабетом — {ratio:.0f} % "
        "від оцінки IDF."
    )
    caveat = "роки різні, а МОЗ рахує всі типи діабету й усі віки"
    if abs(ratio / 100 - 1) <= MOZ_AGREE_TOL:
        return (
            f"{head} Отже, облік МОЗ узгоджується з оцінкою IDF для діагностованих: два "
            "незалежні джерела дають близьку кількість людей із встановленим діагнозом. "
            f"Повного збігу не слід очікувати: {caveat}."
        )
    return f"{head} Розбіжність велика, тому ці числа слід порівнювати обережно: {caveat}."


def _age(indicator: str) -> str:
    found = _AGE.search(indicator)
    return f"{found.group()} років" if found else ""


def _ua_source(r: pd.Series) -> str:
    return f"Україна ({r['source'].split()[0]} {r['year']})"


def _external_row(measure: str, source: str, period: str, r: pd.Series, method: str) -> dict:
    nan = float("nan")
    return {
        "measure": measure, "source": source, "period": period, "age": _age(r["indicator"]),
        "method": method, "value": r["value"], "ci_low": nan, "ci_high": nan,
        "accessed": r["accessed"],
    }  # fmt: skip


def _us_rows(us: pd.DataFrame) -> list[dict]:
    """Оцінки NCHS (загалом, діагностований, недіагностований) — ті, що є у файлі."""
    found = [(us[us["indicator"].str.startswith(p)], m, meth) for p, m, meth in US_ROWS]
    return [
        _external_row(measure, "США (NCHS)", rows.iloc[0]["period"], rows.iloc[0], method)
        for rows, measure, method in found
        if not rows.empty
    ]


def _derived_rows(ukraine: pd.DataFrame) -> list[dict]:
    """Частки, розраховані з кількостей IDF і МОЗ (позначені CALC_NOTE)."""
    counts, rows = ukraine_counts(ukraine), []
    for key, share in ukraine_shares(ukraine).items():
        measure, method = DERIVED[key]
        src = counts[key][1]
        row = _external_row(measure, _ua_source(src), str(src["year"]), src, method)
        row |= {"method": f"{method}; {CALC_NOTE}", "value": share, "age": row["age"] or "усі віки"}
        rows.append(row)
    return rows


def prevalence_comparison(
    nhanes: pd.DataFrame, ukraine: pd.DataFrame, us: pd.DataFrame
) -> pd.DataFrame:
    """Поширеність діабету з різних джерел (колонки COMPARISON_COLUMNS), згрупована за MEASURES.

    nhanes — готові рядки цієї роботи; далі оцінки NCHS, основна оцінка IDF і частки,
    розраховані з кількостей IDF і МОЗ.
    """
    rows = nhanes.to_dict("records") + _us_rows(us)
    if (r := main_prevalence(ukraine)) is not None:
        rows.append(_external_row(MEASURE_TOTAL, _ua_source(r), str(r["year"]), r, IDF_METHOD))
    table = pd.DataFrame(rows + _derived_rows(ukraine), columns=COMPARISON_COLUMNS)
    order = table["measure"].map(MEASURES.index)
    return table.iloc[order.argsort(kind="stable")].reset_index(drop=True)


def other_indicators(ukraine: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """Решта показників обох файлів (колонки US_CSV_COLUMNS, рік → period).

    Без основної оцінки України і без показників NCHS, які вже є в таблиці порівняння.
    """
    i = _main_index(ukraine)
    ua = ukraine.drop(index=[] if i is None else [i]).rename(columns={"year": "period"})
    ua = ua.assign(period=ua["period"].astype(str))
    rest = us[~us["indicator"].str.startswith(tuple(p for p, _, _ in US_ROWS))]
    return pd.concat([ua, rest], ignore_index=True)[US_CSV_COLUMNS]
