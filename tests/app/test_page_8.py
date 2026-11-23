"""Smoke-тест сторінки 8 (експорт) на реальних артефактах."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[2] / "app" / "pages" / "8_📥_Експорт.py"


def test_page_runs_and_lists_model_sheets() -> None:
    at = AppTest.from_file(str(PAGE), default_timeout=180).run()
    assert not at.exception
    assert len(at.get("download_button")) == 2
    text = "\n".join(m.value for m in at.markdown)
    assert "4.2 Метрики моделей" in text
    assert "6.1 США і Україна" in text
    assert [b.label for b in at.get("download_button")] == [
        "Завантажити Excel з усіма результатами",
        "Завантажити всі графіки (ZIP)",
    ]
