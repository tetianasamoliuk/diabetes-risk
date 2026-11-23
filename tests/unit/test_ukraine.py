"""Тести зовнішніх даних по Україні (фікстури вигадані)."""

from pathlib import Path

import pandas as pd
import pytest

from risk.errors import DataValidationError
from risk.external.ukraine import (
    CALC_NOTE,
    COMPARISON_COLS_UK,
    COMPARISON_COLUMNS,
    MEASURE_DIAG,
    MEASURE_TOTAL,
    MEASURE_UNDIAG,
    MEASURES,
    UKRAINE_CSV_COLUMNS,
    US_CSV_COLUMNS,
    load_ukraine_stats,
    load_us_reference,
    main_prevalence,
    moz_idf_text,
    moz_vs_idf,
    other_indicators,
    prevalence_comparison,
    ua_value,
    ukraine_shares,
)


def test_comparison_labels_use_ukrainian_interval() -> None:
    assert COMPARISON_COLS_UK["ci_low"] == "95% ДІ від, %"
    assert COMPARISON_COLS_UK["ci_high"] == "95% ДІ до, %"


HEADER = ",".join(UKRAINE_CSV_COLUMNS)
US_HEADER = ",".join(US_CSV_COLUMNS)
US_BODY = (
    f"{US_HEADER}\nNCHS,2021–2023,Усього діабету серед дорослих ≥ 18 років,15.8,%,u,d\n"
    "NCHS,2021–2023,Діагностований діабет серед дорослих ≥ 18 років,11.3,%,u,d\n"
    ",2021–2023,Недіагностований діабет,4.5,%,u,d\n"
)
UA_BODY = (
    f"{HEADER}\nIDF Atlas,2011,Поширеність діабету 20–79,2.9,%,u,d\n"
    "IDF Atlas,2024,Поширеність діабету 20–79,6.0,%,u,d\n"
    "IDF Atlas,2024,Кількість дорослих з діабетом,100,осіб,u,d\n"
)
# Кількості з тими самими назвами показників, що в data/external/ukraine_stats.csv
UA_COUNTS = (
    f"{HEADER}\nIDF Atlas,2024,Поширеність діабету серед дорослих 20–79,6.0,%,u,d\n"
    "IDF Atlas,2024,Кількість дорослих 20–79 з діабетом (оцінка),800,осіб,u,d\n"
    "IDF Atlas,2024,Кількість дорослих 20–79 з недіагностованим діабетом,300,осіб,u,d\n"
    "IDF Atlas,2024,Доросле населення 20–79,10000,осіб,u,d\n"
    "МОЗ України новина,2025,Пацієнтів із цукровим діабетом в ЕСОЗ,460,осіб,u2,d2\n"
)
NHANES = pd.DataFrame(
    [[MEASURE_TOTAL, "США (NHANES)", "2015–2020", "≥ 20 років", "HbA1c", 13.9, 13.0, 14.8, "—"]],
    columns=COMPARISON_COLUMNS,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "u.csv"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_file(tmp_path: Path) -> None:
    df = load_ukraine_stats(tmp_path / "none.csv")
    assert list(df.columns) == UKRAINE_CSV_COLUMNS
    assert df.empty


def test_empty_zero_bytes_and_header_only(tmp_path: Path) -> None:
    assert load_ukraine_stats(_write(tmp_path, "")).empty
    df = load_ukraine_stats(_write(tmp_path, HEADER + "\n"))
    assert list(df.columns) == UKRAINE_CSV_COLUMNS and df.empty


def test_valid_file(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        f"{HEADER}\nТест,2020,X,5.5,%,http://a,2026-01-01\nТест,2021,Y,7,%,http://a,2026-01-01\n",
    )
    df = load_ukraine_stats(p)
    assert not df.empty
    assert df["year"].dtype.kind == "i" and df["value"].dtype.kind == "f"


def test_wrong_columns(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        load_ukraine_stats(_write(tmp_path, "a,b\n1,2\n"))


def test_non_numeric_value(tmp_path: Path) -> None:
    p = _write(tmp_path, f"{HEADER}\nТест,2020,X,5.5,%,u,d\nТест,2021,Y,abc,%,u,d\n")
    with pytest.raises(DataValidationError, match="рядок 3: value не число"):
        load_ukraine_stats(p)


def test_rows_without_source_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        f"{HEADER}\nТест,2020,X,5.5,%,u,d\n,2020,X,1,%,u,d\nТест,2020,X,2,%,,d\n"
        'Тест,2020,X,3,%,u,\nТест,2020,X,4,%,"  ",d\n',
    )
    df = load_ukraine_stats(p)
    assert list(df["value"]) == [5.5]


def test_us_reference_rows_need_source(tmp_path: Path) -> None:
    us = load_us_reference(_write(tmp_path, US_BODY))
    assert list(us["value"]) == [15.8, 11.3]  # рядок без джерела відкинуто
    assert load_us_reference(tmp_path / "none.csv").empty


def test_prevalence_comparison_rows(tmp_path: Path) -> None:
    ua = load_ukraine_stats(_write(tmp_path, UA_BODY))
    tmp_path.joinpath("us.csv").write_text(US_BODY, encoding="utf-8")
    us = load_us_reference(tmp_path / "us.csv")
    t = prevalence_comparison(NHANES, ua, us)
    # спершу «діабет загалом», далі діагностований; NCHS-рядок без джерела відкинуто
    assert list(t["source"]) == ["США (NHANES)", "США (NCHS)", "Україна (IDF 2024)", "США (NCHS)"]
    assert list(t["measure"]) == [MEASURE_TOTAL] * 3 + [MEASURE_DIAG]
    assert list(t["age"])[1:3] == ["≥ 18 років", "20–79 років"]
    assert t["ci_low"].iloc[1:].isna().all() and t["value"].iloc[2] == 6.0
    empty_us = us.iloc[:0]
    assert list(prevalence_comparison(NHANES, ua.iloc[2:], empty_us)["source"]) == ["США (NHANES)"]


def test_derived_ukraine_rows_and_shares(tmp_path: Path) -> None:
    ua = load_ukraine_stats(_write(tmp_path, UA_COUNTS))
    shares = ukraine_shares(ua)
    assert shares == pytest.approx({"crude": 8.0, "diag": 5.0, "undiag": 3.0, "moz": 4.6})
    t = prevalence_comparison(NHANES, ua, load_us_reference(tmp_path / "none.csv"))
    derived = t[t["method"].str.contains(CALC_NOTE)]
    assert list(derived["value"]) == pytest.approx([8.0, 5.0, 4.6, 3.0])
    assert list(derived["measure"]) == [MEASURE_TOTAL, MEASURE_DIAG, MEASURE_DIAG, MEASURE_UNDIAG]
    moz = derived[derived["source"] == "Україна (МОЗ 2025)"].iloc[0]
    assert moz["age"] == "усі віки" and moz["accessed"] == "d2"
    assert list(t["measure"]) == sorted(t["measure"], key=MEASURES.index)
    assert moz_vs_idf(ua) == pytest.approx(92.0)


def test_derived_need_population_and_counts(tmp_path: Path) -> None:
    ua = load_ukraine_stats(_write(tmp_path, UA_COUNTS))
    no_pop = ua[~ua["indicator"].str.startswith("Доросле")]
    assert ukraine_shares(no_pop) == {}
    no_undiag = ua[~ua["indicator"].str.contains("недіагностованим")]
    assert set(ukraine_shares(no_undiag)) == {"crude", "moz"}
    assert moz_vs_idf(no_undiag) is None and moz_idf_text(no_undiag) is None
    assert ua_value(ua, "Немає такого") is None


def test_moz_idf_text_agrees_or_not(tmp_path: Path) -> None:
    ua = load_ukraine_stats(_write(tmp_path, UA_COUNTS))
    text = moz_idf_text(ua)
    assert "500" in text and "460" in text and "92 %" in text
    assert "узгоджується" in text and "всі типи діабету" in text
    far = ua.assign(value=ua["value"].where(~ua["indicator"].str.startswith("Пацієнтів"), 100.0))
    assert "узгоджується" not in moz_idf_text(far) and "обережно" in moz_idf_text(far)


def test_other_indicators(tmp_path: Path) -> None:
    ua = load_ukraine_stats(_write(tmp_path, UA_BODY))
    tmp_path.joinpath("us.csv").write_text(US_BODY, encoding="utf-8")
    other = other_indicators(ua, load_us_reference(tmp_path / "us.csv"))
    assert list(other.columns) == US_CSV_COLUMNS
    # без основної оцінки України і без показників NCHS, які вже є в порівнянні
    assert list(other["period"]) == ["2011", "2024"]
    assert main_prevalence(ua)["value"] == 6.0
    assert main_prevalence(ua.iloc[2:]) is None
    assert len(other_indicators(ua.iloc[2:], load_us_reference(tmp_path / "none.csv"))) == 1
