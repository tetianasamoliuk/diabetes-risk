"""Схеми: фігура будується, текст не накладається й не виходить за блоки, числа видно."""

import importlib.util
import io
import re
import sys
import tomllib
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image

from risk.data.preprocessing import PREPROCESSING_STEPS
from risk.viz import schemas as s
from risk.viz.style import fig_to_png

ROOT = Path(__file__).resolve().parents[2]
TOL = 1.0  # px: допуск на згладжування контурів
MIN_CYRILLIC = 0.6  # частка кириличних літер серед усіх літер рисунка

_spec = importlib.util.spec_from_file_location(
    "build_schemas", ROOT / "scripts" / "build_schemas.py"
)
bs = importlib.util.module_from_spec(_spec)
sys.modules["build_schemas"] = bs
_spec.loader.exec_module(bs)

MODELS = ["Логістична регресія", "Random Forest", "Gradient Boosting"]
STATUSES = {str(i): "реалізовано" for i in range(1, 10)} | {"2": "частково", "9": "поза обсягом"}
REFS = {str(i): f"Табл. {i}.1–{i}.9" for i in range(1, 10)}
ORIGINAL = [("Вік", "0/2/3/4"), ("ІМТ", "0/1/3"), ("Овочі/фрукти щодня", "0/1"),
            ("Спадковість", "0/3/5")]  # fmt: skip
MODIFIED = [("Вік", "0/2/3/4"), ("ІМТ", "0/1/3"), ("Спадковість", "0/5")]
VARIANTS = [
    {"key": "a", "who": "усі", "glucose": "ні", "target": "ЦД2", "valid": True,
     "n": 2408, "auc": 0.7782},
    {"key": "b", "who": "недіагностовані", "glucose": "так", "target": "ЦД2", "valid": True,
     "n": None, "auc": 0.7536},
    {"key": "c", "who": "усі", "glucose": "так", "target": "ЦД2", "valid": False},
]  # fmt: skip
# Підставні артефакти: тести не залежать від data/processed і models/
FLOW = [
    {"label": "Усі учасники", "n": 25531, "excluded": 0},
    {"label": "З виміряним HbA1c", "n": 13146, "excluded": 1648},
    {"label": "Аналітична когорта", "n": 13140, "excluded": 6},
]
METRICS = {
    "n_train": 10512,
    "n_test": 2628,
    "best_model": "rf",
    "cv": {"rf": {"cv_folds": [0.84, 0.85, 0.83, 0.85, 0.83]}},
    "test": {"logit": {}, "rf": {}, "hgb": {}},
    "findrisc": {
        "a": {"n": 2408, "n_pos": 434, "auc": 0.778, "valid": True},
        "b": {"n": 2031, "n_pos": 60, "auc": 0.754, "valid": True},
        "c": {"n": 2403, "n_pos": 432, "auc": 0.889, "valid": False},
    },
}
NUMBERS = {"n_multivariable": 11304, "n_logit_train": 9071}
NOTES = {"t1d_proxy_n": 73}


@pytest.fixture(autouse=True)
def _close_all() -> Iterator[None]:
    yield
    plt.close("all")


def _texts(fig: Figure) -> list[Text]:
    fig.canvas.draw()
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _plain(fig: Figure) -> str:
    """Увесь текст рисунка одним рядком: переноси і нерозривні пробіли → пробіл."""
    return " ".join(t.get_text() for t in _texts(fig)).replace("\n", " ").replace(s.NBSP, " ")


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _overlap(a: Bbox, b: Bbox) -> bool:
    return a.x0 < b.x1 - TOL and b.x0 < a.x1 - TOL and a.y0 < b.y1 - TOL and b.y0 < a.y1 - TOL


def _inside(a: Bbox, b: Bbox) -> bool:
    return (a.x0 >= b.x0 - TOL and a.x1 <= b.x1 + TOL
            and a.y0 >= b.y0 - TOL and a.y1 <= b.y1 + TOL)  # fmt: skip


def _assert_layout(fig: Figure) -> None:
    """Тексти попарно не перетинаються, не виходять за полотно і за свій блок."""
    texts = _texts(fig)
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    for (a, ta), (b, tb) in combinations(zip(boxes, texts, strict=True), 2):
        assert not _overlap(a, b), (ta.get_text(), tb.get_text())
    frames = [p.get_window_extent(renderer) for p in fig.findobj(FancyBboxPatch)]
    for bb, t in zip(boxes, texts, strict=True):
        assert _inside(bb, fig.bbox), t.get_text()
        x, y = t.get_transform().transform(t.get_position())
        for fr in frames:
            assert not fr.contains(x, y) or _inside(bb, fr), t.get_text()


def _assert_ukrainian(text: str) -> None:
    """Українська мова підписів; назви модулів коду в дужках не враховуються."""
    text = re.sub(r"\([^)]*\)", "", text)
    cyr = len(re.findall(r"[а-яіїєґ]", text, re.IGNORECASE))
    lat = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    assert cyr / (cyr + lat) > MIN_CYRILLIC
    assert re.search(r"[іїєґ]", text, re.IGNORECASE)


def _check(fig: Figure, *phrases: str) -> str:
    """Фігура, валідний PNG, розкладка, українська мова, усі фрази є на рисунку."""
    assert isinstance(fig, Figure)
    assert fig_to_png(fig, dpi=72).startswith(b"\x89PNG")
    _assert_layout(fig)
    text = _plain(fig)
    _assert_ukrainian(text)
    for p in phrases:
        assert p in text, p
    return text


@pytest.mark.parametrize("scope", [(), ("харчування", "CANRISK")])
def test_blocks_map(scope: tuple[str, ...]) -> None:
    text = _check(s.blocks_map(STATUSES, REFS, scope), *scope, "реалізовано", "частково",
                  "поза обсягом", *REFS.values())  # fmt: skip
    for i, (title, pages) in enumerate(s.BLOCKS, 1):
        assert f"{i}. {title}" in text
        assert pages in text
    assert ("Поза обсягом роботи" in text) == bool(scope)


def test_blocks_map_legend_only_used_statuses() -> None:
    text = _check(s.blocks_map(bs.STATUSES, bs.BLOCK_REFS), "реалізовано", "Табл. 3.1–3.7")
    assert "частково" not in text and "поза обсягом" not in text


def test_block_refs_results_tables_range() -> None:
    assert "Табл. 5.1–5.6" in bs.BLOCK_REFS["6"]


def test_out_of_scope_wording() -> None:
    assert bs.OUT_OF_SCOPE[-1] == "PDF-звіт (замість нього — Excel)"
    assert len(bs.OUT_OF_SCOPE) == 6


def test_blocks_map_ukraine_page_in_block_3() -> None:
    assert "Порівняння з Україною" in s.BLOCKS[2][1]
    assert "Україн" not in s.BLOCKS[5][1]


def test_architecture() -> None:
    _check(s.architecture(9), "9 сторінок", "Дані", "Обчислення", "Інтерфейс і звіти",
           "Збережені моделі", "(stats/)", "Потік форми")  # fmt: skip
    names = [name for name, _ in s.COMPUTE_NODES]
    assert "Навчання моделей" in names
    assert "Оцінювання і порівняння моделей" in names
    assert [name for name, _ in s._ui_nodes(9)] == [
        "Excel з таблицями",
        "ZIP графіків PNG",
        "9 сторінок застосунку",
    ]


def test_architecture_ukraine_table_feeds_external_not_stats() -> None:
    data = [name for name, _ in s.DATA_NODES]
    comp = dict(s.COMPUTE_NODES)
    targets = {comp_name for d, c in s.DATA_LINKS
               if data[d] == "Таблиця джерел по Україні"
               for comp_name in [list(comp)[c]]}  # fmt: skip
    assert targets == {"Порівняння з Україною"}
    assert comp["Порівняння з Україною"] == "external/ukraine"
    assert list(comp)[s.SAVE_LINK[0]] == "Навчання моделей"
    assert data[s.SAVE_LINK[1]] == "Збережені моделі"


def test_modelling_pipeline() -> None:
    _check(
        s.modelling_pipeline(13140, 10512, 2628, 5, "Random Forest"),
        _fmt(13140), _fmt(10512), _fmt(2628), "5 фолдах", "Random Forest", "80 %", "20 %",
        "не впливає на вибір моделі", "підборі параметрів, порогів", "Поріг Юдена",
        "тест ДеЛонга", "підібраними параметрами",
    )  # fmt: skip
    text = _plain(s.modelling_pipeline(13140, 10512, 2628, 5, "Random Forest"))
    assert "найкращ" not in text and "out-of-fold" not in text


def test_data_split_all_numbers() -> None:
    _check(
        s.data_split(13140, 10512, 2628, 11304, 9071, 2408, 2031, 60),
        *(_fmt(n) for n in (13140, 10512, 2628, 11304, 9071, 2408, 2031)), "з ЦД2: 60",
        "табл. 3.5", "табл. 4.1", "FINDRISC (a)", "FINDRISC (b)",
    )  # fmt: skip


def test_data_split_missing_numbers_not_drawn() -> None:
    text = _check(s.data_split(13140, 10512, 2628, None, None, None, None, None), "табл. 4.1")
    assert text.count("N =") == 3
    assert "з ЦД2:" not in text


@pytest.mark.parametrize(("no_hba1c", "t1d"), [(1648, 1073), (None, None)])
def test_target_tree(no_hba1c: int | None, t1d: int | None) -> None:
    text = _check(s.target_tree(6, no_hba1c, t1d), "N = 6", "HbA1c ≥ 6.5 %?",
                  "Лікар казав про діабет?", "ЦД2 (діагностований)", "ЦД2 (недіагностований)",
                  "Без ЦД2", "не відокремлюється")  # fmt: skip
    assert (_fmt(1648) in text) == (no_hba1c is not None)
    proxy = f"ймовірний ЦД1 (діагноз до 30 років і інсулін зараз): N = {_fmt(1073)}"
    assert (proxy in text) == (t1d is not None)


def test_findrisc_variants() -> None:
    text = _check(
        s.findrisc_variants(ORIGINAL, MODIFIED, 26, (25, 20), VARIANTS, ["Спадковість: близькі."]),
        "4 пунктів, максимум 26", "3 пунктів, максимум 25", "20 без пункту", "0/1 — прибрано",
        "0/5 (було 0/3/5)", f"N = {_fmt(2408)}", "AUC = 0.778", "AUC = 0.754",
        "витік: лише демонстрація", "придатний для порівняння", "Спадковість: близькі.",
    )  # fmt: skip
    assert text.count("—") >= 2


def test_findrisc_variants_real_items_without_notes() -> None:
    original, modified, max_o, max_m = bs.findrisc_items()
    assert (len(original), len(modified), max_o, max_m) == (8, 7, 26, (25, 20))
    rows = bs.findrisc_variant_rows(METRICS, len(modified), 5)
    _check(s.findrisc_variants(original, modified, max_o, max_m, rows), "(c)")


def test_person_flow() -> None:
    _check(s.person_flow(MODELS, "Gradient Boosting"), *MODELS, "для моделі Gradient Boosting",
           "медіана навчальної вибірки", "лише у FINDRISC", "Бали FINDRISC",
           "Сценарії «що якщо»", "ІМТ зі зросту і ваги")  # fmt: skip


def test_preprocessing_flow() -> None:
    text = _check(s.preprocessing_flow(PREPROCESSING_STEPS))
    for name, what, stage in PREPROCESSING_STEPS:
        assert name in text and what in text and stage in text


@pytest.mark.parametrize(("numbers", "notes"), [(NUMBERS, NOTES), ({}, {})])
def test_build_schemas_saves_nine_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    numbers: dict,
    notes: dict,
) -> None:
    monkeypatch.setattr(bs, "load_artifacts", lambda: (FLOW, METRICS, numbers, notes))
    dirs = [tmp_path / "a", tmp_path / "b"]
    bs.main(["--out", *map(str, dirs)])
    for d in dirs:
        files = sorted(p.name for p in d.glob("*.png"))
        assert len(files) == 9
        assert all((d / f).read_bytes().startswith(b"\x89PNG") for f in files)
    assert ("Увага" in capsys.readouterr().out) == (not notes)


def _map_text() -> str:
    return _plain(s.project_map(bs.PROJECT_MAP_ROWS)).lower()


def test_project_map() -> None:
    fig = s.project_map(bs.PROJECT_MAP_ROWS)
    assert fig_to_png(fig, dpi=72).startswith(b"\x89PNG")
    _assert_layout(fig)
    text = _plain(fig)
    for cell in (*s.MAP_HEADER, *(c for row in bs.PROJECT_MAP_ROWS for c in row)):
        assert cell.replace("\n", " ") in text, cell


def test_project_map_covers_dependencies_and_code_dirs() -> None:
    """Сторож: кожна залежність, тека src/risk/*, app, scripts, tests є на карті."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["dependencies"]
    names = [re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0].lower() for d in deps]
    dirs = [f"src/risk/{p.name}" for p in (ROOT / "src" / "risk").iterdir() if p.is_dir()
            and p.name != "__pycache__"]  # fmt: skip
    text = _map_text()
    for need in (*names, *dirs, "app/", "scripts/", "tests/"):
        assert need in text, need


def test_build_figures_uses_artifact_numbers() -> None:
    figs = bs.build_figures(FLOW, METRICS, NUMBERS, NOTES)
    assert "N = 73" in _plain(figs["схема_цільової_змінної.png"])
    assert _fmt(9071) in _plain(figs["схема_поділу_даних.png"])


def test_build_schemas_missing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bs, "PROCESSED_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Немає"):
        bs.load_artifacts()


def test_build_schemas_help_does_not_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bs, "OUT_DIRS", [tmp_path])
    with pytest.raises(SystemExit) as exc:
        bs.main(["--help"])
    assert exc.value.code == 0
    assert not list(tmp_path.iterdir())


MIN_PRINT_PT = 8.5  # найменший кегль у друці
MIN_MAIN_PT = 9.5  # найменший кегль основних написів блоків у друці
MAX_ASPECT = 1.45  # вища схема впирається у висоту сторінки й зменшується


def _figures() -> dict[str, Figure]:
    return bs.build_figures(FLOW, METRICS, NUMBERS, NOTES)


@pytest.mark.parametrize("name", list(_figures()))
def test_schema_print_size(name: str) -> None:
    """Кегль у друці (ширина друку s.PRINT_WIDTH) і відношення сторін."""
    fig = _figures()[name]
    width, height = fig.get_size_inches()
    smallest = min(t.get_fontsize() for t in _texts(fig))
    assert smallest * s.PRINT_WIDTH / width >= MIN_PRINT_PT, smallest
    assert s.FS * s.PRINT_WIDTH / width >= MIN_MAIN_PT
    assert height / width <= MAX_ASPECT


@pytest.mark.parametrize("name", list(_figures()))
def test_schema_png_width(name: str) -> None:
    """Збережений PNG має ширину полотна (без відступу tight): кегль у друці ≥ MIN_PRINT_PT."""
    fig = _figures()[name]
    smallest = min(t.get_fontsize() for t in _texts(fig))
    png = bs.fig_to_png(fig, dpi=300, tight=False)
    img = Image.open(io.BytesIO(png))
    inches = img.size[0] / img.info["dpi"][0]
    assert inches <= s.WIDTH + 0.02, inches
    assert smallest * s.PRINT_WIDTH / inches >= MIN_PRINT_PT
