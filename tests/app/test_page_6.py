"""Smoke-тест сторінки 6 (порівняння з Україною) через AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[2] / "app" / "pages" / "6_🇺🇦_Порівняння_з_Україною.py"


def run() -> AppTest:
    return AppTest.from_file(str(PAGE), default_timeout=120).run()


def _texts(at: AppTest) -> str:
    return " ".join(e.value for e in [*at.markdown, *at.warning, *at.info, *at.caption])


def test_page_with_data() -> None:
    at = run()
    assert not at.exception
    assert len(at.table) >= 2  # 6.1 і 6.2 — статичні, з перенесенням рядків
    assert "Обмеження" in _texts(at)
    # МОЗ — у таблиці порівняння, з розрахунковими рядками і без «None» в інтервалах
    table = next(t.value for t in at.table if "Показник" in t.value.columns)
    assert "Україна (МОЗ 2025)" in set(table["Джерело"])
    assert not {"None", "nan"} & set(table.astype(str).to_numpy().ravel())
    assert any("узгоджується з оцінкою IDF" in s.value for s in at.success)


def test_page_without_ukraine_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Порожня тека замість data/external → CSV відсутній → шаблон
    monkeypatch.setattr("risk.config.EXTERNAL_DIR", tmp_path)
    at = run()
    assert not at.exception
    notes = [i.value for i in at.info if "не отримані" in i.value]
    assert len(notes) == 1
    assert "перелік перевірених джерел і порожній шаблон таблиці" in notes[0]
    assert not any(ch.isdigit() for ch in notes[0])
    assert len(at.dataframe) >= 1
    assert "Обмеження" in _texts(at)
