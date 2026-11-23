"""Усі сторінки відкриваються без винятків і містять дисклеймер."""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from risk.config import AUTHOR, DISCLAIMER
from risk.errors import ModelNotTrainedError

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = [APP / "Головна.py", *sorted((APP / "pages").glob("*.py"))]
sys.path.insert(0, str(APP))
import _common


def _ukrainian(text: str) -> bool:
    return any("а" <= c.lower() <= "я" or c in "іїєґ" for c in text)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_opens_with_disclaimer(page: Path) -> None:
    at = AppTest.from_file(str(page), default_timeout=240).run()  # сторінка 4 рахує SHAP
    assert not at.exception, [e.value for e in at.exception]
    texts = [e.value for e in [*at.caption, *at.markdown, *at.info, *at.warning]]
    assert not any("Traceback" in t for t in texts)
    assert any(DISCLAIMER in t for t in texts)
    assert any(AUTHOR in t for t in texts)
    # вбудовані написи платформи: один блок стилів із заміною й українські підказки списків
    assert [h.value.count("<style>") for h in at.html] == [1]
    assert all(_ukrainian(w.proto.placeholder) for w in [*at.selectbox, *at.multiselect])


@pytest.fixture
def no_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Збережених моделей немає: завантаження кидає ModelNotTrainedError."""

    def boom(*_: object, **__: object) -> None:
        raise ModelNotTrainedError("немає файлів моделей")

    monkeypatch.setattr(_common, "load_models", boom)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.mark.parametrize("n", [4, 5, 7, 8])
def test_page_without_models_answers_in_ukrainian(no_models: None, n: int) -> None:
    page = next((APP / "pages").glob(f"{n}_*.py"))
    at = AppTest.from_file(str(page), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    assert any("Моделі ще не навчені" in t for t in errors)
    assert any("uv run python scripts/train_models.py" in t for t in errors)
    assert not any("немає файлів моделей" in t for t in errors)
    texts = [e.value for e in [*at.caption, *at.markdown]]
    assert any(DISCLAIMER in t for t in texts)
    assert any(AUTHOR in t for t in texts)


def _cache_decorators(path: Path) -> list[ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found += [d for d in node.decorator_list if "cache_" in ast.unparse(d)]
    return found


@pytest.mark.parametrize("page", [APP / "_common.py", *PAGES], ids=lambda p: p.stem)
def test_every_cache_has_ukrainian_spinner(page: Path) -> None:
    for dec in _cache_decorators(page):
        assert isinstance(dec, ast.Call), ast.unparse(dec)
        spinner = next((k.value for k in dec.keywords if k.arg == "show_spinner"), None)
        assert isinstance(spinner, ast.Constant), ast.unparse(dec)
        assert _ukrainian(str(spinner.value))
