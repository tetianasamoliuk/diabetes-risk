"""Тести валідації користувацького файлу."""

import io
from typing import Never

import pandas as pd
import pytest

from risk.data.user_data import (
    BINARY,
    EXTRA_FIELDS_MSG,
    HEADERS_ONLY_MSG,
    HEADERS_UNKNOWN_MSG,
    LOW_VARIETY_MSG,
    RANGES,
    READ_FAILED_MSG,
    REQUIRED_COLUMNS,
    TEMPLATE_COLUMNS,
    XLSX_FAILED_MSG,
    complete_cases_shortfall,
    template_csv,
    template_description,
    template_xlsx,
    validate_user_file,
)


def _good(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [40 + i for i in range(n)],
            "male": [i % 2 for i in range(n)],
            "bmi": 25.0,
            "waist": 90.0,
            "t2d": [i % 2 for i in range(n)],
        }
    )


def _csv(df: pd.DataFrame, sep: str = ",") -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8")


def test_template_roundtrip() -> None:
    raw = template_csv()
    assert raw.startswith(b"\xef\xbb\xbf")
    # 3 рядки прикладів < 10: дублюємо до 18 і перевіряємо повну валідність
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    big = pd.concat([df] * 6, ignore_index=True)
    out, errs = validate_user_file(_csv(big), "t.csv")
    assert errs == [] and out is not None
    assert list(out.columns[:12]) == list(TEMPLATE_COLUMNS)
    assert (out["pa_min_week"] == out["pa_h_week"] * 60).all()
    assert (out["cycle_2017"] == 0).all() and out["diq010"].isna().all()


def test_template_too_short() -> None:
    out, errs = validate_user_file(template_csv(), "t.csv")
    assert out is None and errs == ["Замало рядків для аналізу: 3, потрібно щонайменше 10"]


def test_description() -> None:
    d = template_description()
    assert len(d) == 12
    assert set(d.loc[d["Обов'язкова"] == "так", "Колонка"]) == set(REQUIRED_COLUMNS)


def test_missing_column_and_optional_nan() -> None:
    out, errs = validate_user_file(_csv(_good().drop(columns="bmi")), "a.csv")
    assert out is None and 'Немає обов\'язкової колонки "bmi"' in errs
    out, errs = validate_user_file(_csv(_good()), "a.csv")
    assert errs == [] and out["sbp"].isna().all()


def test_text_in_numeric_row_number() -> None:
    df = _good().astype({"age": object})
    df.loc[13 - 2, "age"] = "abc"  # рядок 13 у файлі
    _, errs = validate_user_file(_csv(df), "a.csv")
    assert 'У колонці "age" є текст у рядку 13' in errs


def test_out_of_range() -> None:
    df = _good()
    df.loc[1, "age"] = 250
    out, errs = validate_user_file(_csv(df), "a.csv")
    assert out is None and "Вік 250 поза допустимим діапазоном 20–120 (рядок 3)" in errs


def test_binary_and_t2d_missing() -> None:
    df = _good()
    df.loc[2, "male"] = 2
    df.loc[3, "t2d"] = None
    _, errs = validate_user_file(_csv(df), "a.csv")
    assert any(
        e.startswith('У колонці "male" допустимі лише 0 або 1; інші значення в рядку 4')
        for e in errs
    )
    assert 'У колонці "t2d" є пропуск у рядку 5' in errs


def test_empty_and_bad_format() -> None:
    assert validate_user_file(b"", "a.csv") == (None, ["Файл порожній"])
    assert validate_user_file(b"age,male,bmi,waist,t2d\n", "a.csv")[1] == [HEADERS_ONLY_MSG]
    assert validate_user_file(b"x", "a.txt") == (None, ["Підтримуються лише .csv і .xlsx"])


def test_xlsx_and_semicolon() -> None:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _good().to_excel(w, index=False)
    out, errs = validate_user_file(buf.getvalue(), "a.XLSX")
    assert errs == [] and len(out) == 12
    out, errs = validate_user_file(_csv(_good(), ";"), "a.csv")
    assert errs == [] and len(out) == 12


def test_broken_xlsx() -> None:
    out, errs = validate_user_file(b"not zip", "a.xlsx")
    assert out is None and errs == [XLSX_FAILED_MSG]


def test_fully_empty_required_column_is_one_message() -> None:
    df = _good(40)
    df["bmi"] = None
    out, errs = validate_user_file(_csv(df), "a.csv")
    assert out is None and errs == ['Колонка "bmi" повністю порожня']


def test_unknown_headers_get_one_hint() -> None:
    df = pd.DataFrame({"вік": range(12), "імт": range(12)})
    _, errs = validate_user_file(_csv(df), "a.csv")
    assert errs.count(HEADERS_UNKNOWN_MSG) == 1 and len(errs) == len(REQUIRED_COLUMNS) + 1
    assert "шаблон" in HEADERS_UNKNOWN_MSG
    _, errs = validate_user_file(_csv(_good().drop(columns="bmi")), "a.csv")
    assert HEADERS_UNKNOWN_MSG not in errs


def test_error_cap() -> None:
    df = _good(40)
    df["age"] = 500
    _, errs = validate_user_file(_csv(df), "a.csv")
    assert len(errs) == 21 and errs[-1] == "… та ще 20 помилок"


def test_real_cohort_extremes_are_in_range() -> None:
    df = _good().assign(tc=[545.0] * 2 + [250.0] * 10, hdl=[5.0, 226.0] + [50.0] * 10)
    df["bmi"] = [92.3] + [25.0] * 11
    out, errs = validate_user_file(_csv(df), "a.csv")
    assert errs == [] and out is not None


def test_error_cap_plural_forms() -> None:
    def last(n_rows: int) -> str:
        df = _good(n_rows).assign(age=500)
        return validate_user_file(_csv(df), "a.csv")[1][-1]

    assert last(21) == "… та ще 1 помилка"
    assert last(22) == "… та ще 2 помилки"
    assert last(25) == "… та ще 5 помилок"


def test_decimal_comma_semicolon_csv() -> None:
    df = _good().astype({"bmi": object, "waist": object})
    df["bmi"] = ["27,5"] * 11 + ["30"]  # ціле без коми в тій самій колонці
    df["waist"] = "90,5"
    out, errs = validate_user_file(_csv(df, ";"), "a.csv")
    assert errs == [] and out["bmi"].iloc[0] == 27.5 and out["waist"].iloc[0] == 90.5


def test_decimal_comma_does_not_hide_text() -> None:
    df = _good().astype({"bmi": object})
    df.loc[0, "bmi"] = "abc"
    df.loc[1, "bmi"] = "27,5"
    _, errs = validate_user_file(_csv(df, ";"), "a.csv")
    assert 'У колонці "bmi" є текст у рядку 2' in errs


def test_column_names_case_and_spaces() -> None:
    df = _good().rename(columns={"age": " AGE ", "t2d": "T2D"})
    out, errs = validate_user_file(_csv(df), "a.csv")
    assert errs == [] and out["age"].iloc[0] == 40


def test_duplicate_columns() -> None:
    raw = b"age,male,bmi,waist,t2d,age\n" + b"40,1,25,90,1,41\n" * 12
    assert validate_user_file(raw, "a.csv") == (None, ['Колонка "age" трапляється двічі'])
    raw = b"age,male,bmi,waist,t2d,AGE\n" + b"40,1,25,90,1,41\n" * 12
    assert validate_user_file(raw, "a.csv")[1] == ['Колонка "age" трапляється двічі']


def test_xlsx_with_text_content_is_ukrainian_message() -> None:
    assert validate_user_file(_csv(_good()), "a.xlsx") == (None, [XLSX_FAILED_MSG])


def test_t2d_needs_both_classes() -> None:
    msg = 'У колонці "t2d" мають бути і 0, і 1 (щонайменше по 5 спостережень)'
    one = _good(12).assign(t2d=0)
    assert msg in validate_user_file(_csv(one), "a.csv")[1]
    few = _good(12).assign(t2d=[1, 1] + [0] * 10)
    assert msg in validate_user_file(_csv(few), "a.csv")[1]
    assert validate_user_file(_csv(_good(12)), "a.csv")[1] == []


# ---------- крайні випадки: людські повідомлення ----------
def _human(errs: list[str]) -> None:
    # Жодних трейсбеків і назв винятків Python у повідомленнях
    text = " ".join(errs)
    for bad in ("Traceback", "Error", "Exception", 'File "', "UnicodeDecode", "codec"):
        assert bad not in text, text


def test_binary_garbage_as_csv_is_human_readable() -> None:
    out, errs = validate_user_file(bytes(range(256)) * 4, "a.csv")
    assert out is None and errs
    _human(errs)


def test_no_required_columns_at_all() -> None:
    df = pd.DataFrame({"foo": range(12), "bar": range(12)})
    out, errs = validate_user_file(_csv(df), "a.csv")
    assert out is None
    assert {f'Немає обов\'язкової колонки "{c}"' for c in REQUIRED_COLUMNS} <= set(errs)
    _human(errs)


def test_text_file_extension_message() -> None:
    out, errs = validate_user_file(b"hello world", "notes.txt")
    assert out is None and errs == ["Підтримуються лише .csv і .xlsx"]
    _human(errs)


def test_csv_parser_error_is_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    import risk.data.user_data as ud

    def boom(buf: io.BytesIO, name: str) -> Never:
        raise pd.errors.ParserError("Error tokenizing data")

    monkeypatch.setattr(ud, "_read_table", boom)
    out, errs = validate_user_file(b"a,b\n1,2\n", "a.csv")
    assert out is None and errs[0].startswith("Не вдалося прочитати файл")
    _human(errs)


def test_broken_xlsx_message_is_human_readable() -> None:
    _, errs = validate_user_file(b"not zip", "a.xlsx")
    _human(errs)


def test_out_of_range_and_text_messages_are_human_readable() -> None:
    df = _good().astype({"bmi": object})
    df.loc[0, "age"] = 5
    df.loc[1, "bmi"] = "abc"
    _, errs = validate_user_file(_csv(df), "a.csv")
    assert errs
    _human(errs)


def test_empty_data_error_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kw: object) -> pd.DataFrame:
        raise pd.errors.EmptyDataError("No columns to parse")

    monkeypatch.setattr(pd, "read_csv", boom)
    df, errors = validate_user_file(b"a,b\n1,2\n", "x.csv")
    assert df is None
    assert errors == ["Файл порожній"]


def test_every_numeric_column_has_range() -> None:
    assert set(TEMPLATE_COLUMNS) - set(BINARY) == set(RANGES)


# ---------- атаки на файл ----------
COLUMNS = "age,male,bmi,waist,sbp,dbp,tc,hdl,pa_h_week,fam_dm,bp_med,t2d"


def _rows(n: int = 30, **override: object) -> pd.DataFrame:
    """Валідний набір з усіма колонками шаблону; override замінює колонки."""
    i = pd.Series(range(n))
    base = {
        "age": 30 + i % 40, "male": i % 2, "bmi": 22 + i % 9, "waist": 80 + i % 25,
        "sbp": 110 + i % 30, "dbp": 70 + i % 15, "tc": 170 + i % 40, "hdl": 40 + i % 20,
        "pa_h_week": 1 + i % 5, "fam_dm": i % 3 % 2, "bp_med": i % 4 % 2, "t2d": i // 3 % 2,
    }  # fmt: skip
    return pd.DataFrame({**base, **override})


def test_valid_full_set_is_accepted() -> None:
    out, errs = validate_user_file(_csv(_rows()), "a.csv")
    assert errs == [] and len(out) == 30


def test_trailing_comma_in_rows_is_ignored_not_shifted() -> None:
    lines = _csv(_rows()).decode().splitlines()
    raw = "\n".join([lines[0], *[f"{line}," for line in lines[1:]]]).encode()
    out, errs = validate_user_file(raw, "a.csv")
    assert errs == [] and out["age"].iloc[0] == 30 and out["t2d"].iloc[0] == 0


def test_two_trailing_commas_are_reported() -> None:
    lines = _csv(_rows()).decode().splitlines()
    raw = "\n".join([lines[0], *[f"{line},," for line in lines[1:]]]).encode()
    assert validate_user_file(raw, "a.csv") == (None, [EXTRA_FIELDS_MSG])


def test_leading_id_column_without_header_is_reported() -> None:
    lines = _csv(_rows()).decode().splitlines()
    raw = "\n".join([lines[0], *[f"id{i},{line}" for i, line in enumerate(lines[1:])]]).encode()
    assert validate_user_file(raw, "a.csv") == (None, [EXTRA_FIELDS_MSG])


@pytest.mark.parametrize("error", [pd.errors.ParserError, KeyError, TypeError, IndexError])
def test_unexpected_reader_errors_become_one_ukrainian_message(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    def boom(buf: io.BytesIO, name: str) -> Never:
        raise error("x")

    monkeypatch.setattr("risk.data.user_data._read_table", boom)
    assert validate_user_file(b"a,b\n1,2\n", "a.csv") == (None, [READ_FAILED_MSG])


def test_cp1251_file_is_not_utf8_message() -> None:
    raw = "вік,стать\n40,1\n".encode("cp1251") * 3
    out, errs = validate_user_file(raw, "a.csv")
    assert out is None and "UTF-8" in errs[0]


def test_yes_no_in_binary_column_is_one_message_with_hint() -> None:
    out, errs = validate_user_file(_csv(_rows(t2d=["yes", "no"] * 15)), "a.csv")
    assert out is None and len(errs) == 1
    assert errs[0].startswith('У колонці "t2d" допустимі лише 0 або 1; інші значення в рядках')
    assert "2, 3, 4 … (ще 27)" in errs[0] and "1/0" in errs[0]


def test_binary_message_lists_up_to_three_rows_and_singular() -> None:
    few = _rows(t2d=[0, 1, 5, 7] * 7 + [0, 1])
    errs = validate_user_file(_csv(few), "a.csv")[1]
    assert "в рядках 4, 5, 8 … (ще 11)" in errs[0]
    one = _rows(t2d=[0, 1, 2] + [0, 1] * 13 + [1])
    assert "в рядку 4. " in validate_user_file(_csv(one), "a.csv")[1][0]


def test_sbp_not_above_dbp_is_error_per_row() -> None:
    out, errs = validate_user_file(_csv(_rows(sbp=90, dbp=100)), "a.csv")
    assert out is None and errs[0] == "Систолічний тиск має бути вищим за діастолічний (рядок 2)"
    equal = validate_user_file(_csv(_rows(sbp=100, dbp=100)), "a.csv")[1]
    assert equal and "вищим за діастолічний" in equal[0]


def test_total_cholesterol_below_hdl_is_error() -> None:
    out, errs = validate_user_file(_csv(_rows(tc=100, hdl=200)), "a.csv")
    assert out is None and errs[0] == "Загальний холестерин не може бути меншим за ЛПВЩ (рядок 2)"


def test_headers_only_message() -> None:
    assert validate_user_file(COLUMNS.encode() + b"\n", "a.csv") == (None, [HEADERS_ONLY_MSG])


def test_all_numeric_columns_constant_is_rejected() -> None:
    const = _rows(age=40, bmi=27, waist=90, sbp=120, dbp=80, tc=190, hdl=50, pa_h_week=3)
    assert validate_user_file(_csv(const), "a.csv") == (None, [LOW_VARIETY_MSG])


def test_optional_columns_filled_in_different_rows_is_rejected() -> None:
    sparse = _rows()
    sparse.loc[::2, "sbp"] = None
    sparse.loc[1::2, "tc"] = None
    assert validate_user_file(_csv(sparse), "a.csv") == (None, [LOW_VARIETY_MSG])


def test_tiny_set_with_empty_optional_column_is_accepted() -> None:
    tiny = _rows(10, sbp=None, t2d=[0, 1] * 5)
    out, errs = validate_user_file(_csv(tiny), "a.csv")
    assert errs == [] and len(out) == 10


def test_ranges_are_human_readable() -> None:
    huge = _rows(age=[1e9, float("inf"), float("-inf"), -5.5] + [40] * 26)
    errs = validate_user_file(_csv(huge), "a.csv")[1]
    assert "Вік 1 000 000 000 поза допустимим діапазоном 20–120 (рядок 2)" in errs
    assert "Вік ∞ поза допустимим діапазоном 20–120 (рядок 3)" in errs
    assert "Вік −∞ поза допустимим діапазоном 20–120 (рядок 4)" in errs
    assert "Вік −5.5 поза допустимим діапазоном 20–120 (рядок 5)" in errs
    assert not any("e+" in e or "inf" in e for e in errs)


def test_xlsx_template_roundtrip_and_sheets() -> None:
    sheets = pd.read_excel(io.BytesIO(template_xlsx()), sheet_name=None)
    assert list(sheets) == ["Дані", "Опис колонок"]
    assert list(sheets["Дані"].columns) == list(TEMPLATE_COLUMNS) and len(sheets["Дані"]) == 3
    assert len(sheets["Опис колонок"]) == 12


def test_xlsx_with_extra_blank_column_is_accepted() -> None:
    buf = io.BytesIO()
    _rows().assign(**{"": None}).to_excel(buf, index=False)
    out, errs = validate_user_file(buf.getvalue(), "a.xlsx")
    assert errs == [] and len(out) == 30


def test_complete_cases_shortfall() -> None:
    df = _rows(60)
    assert complete_cases_shortfall(df, ["age", "bmi"]) is None
    msg = complete_cases_shortfall(df.head(20), ["age", "bmi"])
    assert msg and "потрібно щонайменше 30, є 20" in msg
