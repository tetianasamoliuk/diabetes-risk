"""Схеми (діаграми) для документів і застосунку. Кожна функція повертає matplotlib Figure.

Полотно схеми має координати в дюймах, тому ширина блоків і кегль узгоджуються напряму.
Рядки переносяться за реальною шириною тексту шрифту (TextPath), а не за кількістю символів.
Числа результатів приходять лише параметрами.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import TypedDict, Unpack

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.textpath import TextPath

from risk.config import HBA1C_THRESHOLD
from risk.viz.style import PALETTE, apply_style

type Point = tuple[float, float]
type Item = tuple[str, str]

PT = 1 / 72  # дюйма в одному пункті
FS = 10.0
FS_SMALL = 9.0
LINE_EM = 1.25  # висота рядка в кеглях (DejaVu Sans, міжрядковий інтервал matplotlib)
PRINT_WIDTH = 17 / 2.54  # дюйми: текстове поле A4 з полями 2 см
WIDTH = 7.0  # полотно, дюйми: у PDF масштаб 0.96 (PRINT_WIDTH / WIDTH), 10 → 9.6 pt, 9 → 8.6 pt
MARGIN = 0.1
PAD = 0.08
GAP = 0.05
RADIUS = 0.06
ARROW_GAP = 0.3
TITLE_H = 0.35
FRAME_TITLE_H = 2 * FS * LINE_EM * PT + 2 * PAD
RENDER_SLACK = 1.07  # рендер ширший за контур гліфів (бокові поля, хінтинг)
TAG_OFFSET = 0.08
NBSP = "\u00a0"

INK = "#262624"
MUTED = "#55544f"
EDGE = str(PALETTE["neutral"])
BLUE = str(PALETTE["t2d0"])
ORANGE = str(PALETTE["t2d1"])
GREEN = list(PALETTE["models"])[2]
PURPLE = str(PALETTE["accent"])
RED = "#c62f2b"


@dataclass(frozen=True, slots=True)
class Look:
    """Вигляд рамки блоку."""

    fc: str = "#ffffff"
    ec: str = EDGE
    ls: str = "-"
    lw: float = 1.0


PLAIN = Look()
BAND = Look("#f7f7f5", "#c9c8c3")
DATA = Look("#eaf1fb", BLUE)
TRAIN = Look("#f3f7fd", BLUE, lw=1.4)
TEST = Look("#fdf3ee", ORANGE, "--", 1.8)
T2D = Look("#fdeee6", ORANGE)
NO_T2D = Look("#eaf1fb", BLUE)
EXCLUDED = Look("#f0f0ee", EDGE, "--")
QUESTION = Look("#ffffff", PURPLE, lw=1.4)
SIDE = Look("#ffffff", EDGE, "--")
FORM = Look("#eeebfb", PURPLE)
HEADER = Look("#f0f0ee", EDGE)
VALID = Look("#e4f5ee", GREEN)
LEAK = Look("#fbe4e2", RED, lw=1.8)
STAGE_LOOKS = (DATA, Look("#e4f5ee", GREEN), FORM, T2D)
OUT_OF_SCOPE = "поза обсягом"
STATUS_LOOKS = {
    "реалізовано": Look("#e4f5ee", GREEN),
    "частково": Look("#fdeee6", ORANGE),
    OUT_OF_SCOPE: Look("#f0f0ee", EDGE, "--"),
}


@dataclass(frozen=True, slots=True)
class Box:
    """Прямокутник на полотні: центр, ширина, висота (дюйми)."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def top(self) -> Point:
        """Середина верхньої сторони (x, y)."""
        return self.cx, self.cy + self.h / 2

    @property
    def bottom(self) -> Point:
        """Середина нижньої сторони (x, y)."""
        return self.cx, self.cy - self.h / 2

    @property
    def left(self) -> Point:
        """Середина лівої сторони (x, y)."""
        return self.cx - self.w / 2, self.cy

    @property
    def right(self) -> Point:
        """Середина правої сторони (x, y)."""
        return self.cx + self.w / 2, self.cy


class LineStyle(TypedDict, total=False):
    """Необов'язковий стиль ламаної і стрілки: колір, тип і товщина лінії."""

    color: str
    ls: str
    lw: float


# ---------- примітиви ----------


@lru_cache(maxsize=8192)
def _width(s: str, fs: float, bold: bool) -> float:
    """Ширина рядка в дюймах за контуром гліфів."""
    prop = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=fs, prop=prop).get_extents().width * PT * RENDER_SLACK


def _wrap_line(line: str, width: float, fs: float, bold: bool) -> str:
    out: list[str] = []
    for word in line.split(" "):
        joined = f"{out[-1]} {word}" if out else word
        if out and _width(joined, fs, bold) <= width:
            out[-1] = joined
        else:
            out.append(word)
    return "\n".join(out)


def _wrap(text: str, width: float, fs: float = FS, bold: bool = False) -> str:
    """Перенос за словами так, щоб кожен рядок вміщався в width дюймів."""
    return "\n".join(_wrap_line(line, width, fs, bold) for line in text.split("\n"))


def _text_h(text: str, fs: float) -> float:
    return (text.count("\n") + 1) * fs * LINE_EM * PT


def _n(n: int) -> str:
    """Ціле з нерозривним пробілом між тисячами: 13 140."""
    return f"{n:,}".replace(",", NBSP)


def _n_line(label: str, n: int | None) -> str:
    """Підпис і рядок «N = …»; без числа — лише підпис."""
    return label if n is None else f"{label}\nN = {_n(n)}"


def _box_h(w: float, text: str, note: str = "", bold: bool = False) -> float:
    """Висота блоку, у який уміщаються текст і дрібна примітка."""
    inner = w - 2 * PAD
    h = _text_h(_wrap(text, inner, FS, bold), FS)
    if note:
        h += GAP + _text_h(_wrap(note, inner, FS_SMALL), FS_SMALL)
    return h + 2 * PAD


def _canvas(h: float, w: float = WIDTH) -> tuple[Figure, Axes]:
    """Порожнє полотно w × h дюймів з координатами в дюймах."""
    apply_style()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _frame(ax: Axes, box: Box, look: Look) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (box.cx - box.w / 2, box.cy - box.h / 2),
            box.w,
            box.h,
            boxstyle=f"round,pad=0,rounding_size={RADIUS}",
            fc=look.fc,
            ec=look.ec,
            ls=look.ls,
            lw=look.lw,
        )
    )


def _label(ax: Axes, box: Box, text: str, note: str, bold: bool) -> None:
    """Основний текст і дрібна примітка під ним, разом по центру блоку."""
    inner = box.w - 2 * PAD
    body = _wrap(text, inner, FS, bold)
    hb = _text_h(body, FS)
    small = _wrap(note, inner, FS_SMALL) if note else ""
    hn = _text_h(small, FS_SMALL) + GAP if note else 0.0
    top = box.cy + (hb + hn) / 2
    _tag(ax, (box.cx, top - hb / 2), body, ha="center", color=INK, bold=bold, fs=FS)
    if note:
        _tag(ax, (box.cx, top - hb - (hn + GAP) / 2), small, ha="center")


def _box(
    ax: Axes,
    cx: float,
    cy: float,
    w: float,
    text: str,
    *,
    note: str = "",
    look: Look = PLAIN,
    bold: bool = False,
    h: float | None = None,
) -> Box:
    """Блок із центром (cx, cy); висота за текстом, якщо не задана."""
    box = Box(cx, cy, w, _box_h(w, text, note, bold) if h is None else h)
    _frame(ax, box, look)
    _label(ax, box, text, note, bold)
    return box


def _tag(
    ax: Axes,
    xy: Point,
    text: str,
    *,
    ha: str = "left",
    color: str = MUTED,
    bold: bool = False,
    fs: float | None = None,
) -> None:
    """Підпис із центром по вертикалі; кегль за замовчуванням: жирний — FS, інакше FS_SMALL."""
    size = fs if fs is not None else (FS if bold else FS_SMALL)
    weight = "bold" if bold else "normal"
    ax.text(*xy, text, ha=ha, va="center", fontsize=size, color=color, fontweight=weight)


def _arrow(
    ax: Axes, start: Point, end: Point, *, color: str = EDGE, ls: str = "-", lw: float = 1.2
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": lw,
            "linestyle": ls,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 12,
        },
    )


def _route(ax: Axes, points: Sequence[Point], **style: Unpack[LineStyle]) -> None:
    """Ламана зі стрілкою на останньому відрізку."""
    xs, ys = zip(*points[:-1], strict=True)
    ax.plot(
        xs, ys, color=style.get("color", EDGE), ls=style.get("ls", "-"), lw=style.get("lw", 1.2)
    )
    _arrow(ax, points[-2], points[-1], **style)


def _column(
    ax: Axes,
    items: Sequence[Item],
    heights: Sequence[float],
    cx: float,
    w: float,
    top: float,
    *,
    look: Look = PLAIN,
    bold: bool = False,
    gap: float = ARROW_GAP,
) -> list[Box]:
    """Стовпчик блоків (текст, примітка) зверху вниз, зі стрілками між сусідніми."""
    boxes: list[Box] = []
    for (text, note), h in zip(items, heights, strict=True):
        boxes.append(_box(ax, cx, top - h / 2, w, text, note=note, look=look, bold=bold, h=h))
        if len(boxes) > 1:
            _arrow(ax, boxes[-2].bottom, boxes[-1].top)
        top -= h + gap
    return boxes


def _legend(ax: Axes, looks: dict[str, Look]) -> None:
    handles = [Patch(fc=lk.fc, ec=lk.ec, ls=lk.ls, label=name) for name, lk in looks.items()]
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=FS,
        bbox_to_anchor=(0.5, 0.0),
    )


# ---------- 1. Дев'ять блоків керівника ----------

BLOCKS: list[Item] = [
    ("Збір та завантаження даних", "Стор. 1 «Дані і вибірка»"),
    ("Попередня обробка", "Стор. 1 «Дані і вибірка»"),
    (
        "Статистичне дослідження вибірки",
        "Стор. 2 «Статистика»; стор. 6 «Порівняння з Україною» — підтвердження існуючими даними",
    ),
    ("Аналіз факторів ризику", "Стор. 3 «Фактори ризику»"),
    ("Математичне моделювання", "Стор. 4 «Моделі»"),
    ("Порівняння моделей-аналогів", "Стор. 5 «FINDRISC»"),
    ("Перевірка та оцінювання моделі", "Стор. 4 «Моделі»"),
    ("Оцінювання індивідуального ризику", "Стор. 7 «Оцінка ризику»"),
    ("Візуалізація та формування звіту", "Стор. 8 «Експорт» і кнопки PNG під кожним графіком"),
]  # (назва блоку керівника, сторінки застосунку)
BLOCK_W = 2.55
PAGES_W = WIDTH - 2 * MARGIN - BLOCK_W - 0.25
LEGEND_H = 0.4


def _block_h(i: int, block: tuple[str, str, str]) -> float:
    title, pages, items = block
    return max(_box_h(BLOCK_W, f"{i}. {title}", bold=True), _box_h(PAGES_W, pages, items))


def _block_row(
    ax: Axes, i: int, block: tuple[str, str, str], cy: float, h: float, look: Look
) -> Box:
    """Блок керівника ліворуч і його сторінки й таблиці праворуч."""
    title, pages, items = block
    left = _box(ax, MARGIN + BLOCK_W / 2, cy, BLOCK_W, f"{i}. {title}", look=look, bold=True, h=h)
    right = _box(ax, WIDTH - MARGIN - PAGES_W / 2, cy, PAGES_W, pages, note=items, h=h)
    ax.plot([left.right[0], right.left[0]], [cy, cy], color=EDGE, lw=0.8, ls=":")
    return left


def blocks_map(
    statuses: dict[str, str], refs: dict[str, str], out_of_scope: Sequence[str] = ()
) -> Figure:
    """9 блоків керівника в порядку стрілок; поруч сторінки застосунку; колір — статус.

    statuses: номер блоку ("1"…"9") → «реалізовано» | «частково» | «поза обсягом».
    refs: номер блоку → ключові таблиці й рисунки (рядок під сторінками).
    out_of_scope: пункти вимог поза обсягом роботи (окремий блок унизу).
    """
    blocks = [(title, pages, refs[str(i)]) for i, (title, pages) in enumerate(BLOCKS, 1)]
    heights = [_block_h(i, b) for i, b in enumerate(blocks, 1)]
    scope = "Поза обсягом роботи: " + ", ".join(out_of_scope) if out_of_scope else ""
    scope_h = _box_h(WIDTH - 2 * MARGIN, scope) if scope else 0.0
    total = sum(heights) + ARROW_GAP * len(blocks) + scope_h + LEGEND_H + 2 * MARGIN
    fig, ax = _canvas(total)
    y, prev = total - MARGIN, None
    for i, (block, h) in enumerate(zip(blocks, heights, strict=True), 1):
        left = _block_row(ax, i, block, y - h / 2, h, STATUS_LOOKS[statuses[str(i)]])
        if prev is not None:
            _arrow(ax, prev.bottom, left.top)
        prev, y = left, y - h - ARROW_GAP
    if scope:
        look = STATUS_LOOKS[OUT_OF_SCOPE]
        _box(ax, WIDTH / 2, y - scope_h / 2, WIDTH - 2 * MARGIN, scope, look=look, h=scope_h)
    used = set(statuses.values()) | ({OUT_OF_SCOPE} if scope else set())
    _legend(ax, {name: lk for name, lk in STATUS_LOOKS.items() if name in used})
    return fig


# ---------- 2. Архітектура ----------

DATA_NODES: list[Item] = [
    ("Файли NHANES (CDC)", "data/download, data/load"),
    ("Файл користувача CSV або Excel", "data/user_data"),
    ("Таблиця джерел по Україні", "data/external/\nukraine_stats.csv"),
    ("Збережені моделі", "models/*.joblib, metrics.json"),
]
COMPUTE_NODES: list[Item] = [
    ("Підготовка даних", "data/harmonize, data/cohort"),
    ("Статистичний аналіз", "stats/"),
    ("Порівняння з Україною", "external/ukraine"),
    ("Навчання моделей", "models/train, models/pipelines"),
    ("Оцінювання і порівняння моделей", "models/evaluate, models/delong, models/summary"),
    ("Шкала FINDRISC", "findrisc/score"),
    ("Пояснення результатів: SHAP і автотексти", "models/explain, interpret"),
]
DATA_LINKS = ((0, 0), (1, 0), (2, 2))  # (блок «Дані», блок «Обчислення»)
SAVE_LINK = (3, 3)  # «Навчання моделей» → «Збережені моделі»
COL_GAP = 0.4
COL_W = (WIDTH - 2 * MARGIN - 2 * COL_GAP) / 3
STACK_GAP = 0.18
ROUTE_DROP = 0.25
FORM_ROUTE = (
    "Потік форми «Оцінка ризику»: значення людини → медіани замість «не знаю» → "
    "збережені моделі → ймовірності і SHAP"
)


def _ui_nodes(n_pages: int) -> list[Item]:
    return [
        ("Excel з таблицями", "report/excel"),
        ("ZIP графіків PNG", "report/excel, viz/figures"),
        (f"{n_pages} сторінок застосунку", "app/, Streamlit"),
    ]


def _node_h(nodes: list[Item]) -> float:
    """Однакова висота блоків одного рівня — за найвищим."""
    return max(_box_h(COL_W - 2 * PAD, name, f"({module})") for name, module in nodes)


def _tier(ax: Axes, i: int, title: str, nodes: list[Item], top: float, band_h: float) -> list[Box]:
    """Рівень системи: рамка-стовпчик із заголовком і блоками, рівномірно по висоті."""
    cx, box_h = MARGIN + COL_W / 2 + i * (COL_W + COL_GAP), _node_h(nodes)
    _frame(ax, Box(cx, top - band_h / 2, COL_W, band_h), BAND)
    _tag(ax, (cx, top - TITLE_H / 2), title, ha="center", color=INK, bold=True)
    first, last = top - TITLE_H - box_h / 2, top - band_h + PAD + box_h / 2
    step = (first - last) / (len(nodes) - 1)
    w = COL_W - 2 * PAD
    return [
        _box(ax, cx, first - k * step, w, name, note=f"({module})", look=DATA, h=box_h)
        for k, (name, module) in enumerate(nodes)
    ]


def _tier_links(ax: Axes, data: list[Box], comp: list[Box], ui: list[Box]) -> None:
    for d, c in DATA_LINKS:
        _arrow(ax, data[d].right, comp[c].left)
    src, dst = SAVE_LINK
    _arrow(ax, comp[src].left, data[dst].right, color=BLUE, ls="--")
    mid = (comp[0].cy + comp[-1].cy) / 2
    ax.annotate(
        "",
        xy=(ui[0].left[0] - 2 * PAD, mid),
        xytext=(comp[0].right[0] + 2 * PAD, mid),
        arrowprops={
            "arrowstyle": "simple,head_width=1.4,head_length=0.9,tail_width=0.6",
            "fc": "#d9d8d3",
            "ec": EDGE,
            "mutation_scale": 14,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def _form_route(ax: Axes, start: Box, end: Box, y: float) -> None:
    _route(ax, [start.bottom, (start.cx, y), (end.cx, y), end.bottom], color=PURPLE, lw=1.6)
    text = _wrap(FORM_ROUTE, WIDTH - 2 * MARGIN, FS_SMALL)
    _tag(
        ax,
        (WIDTH / 2, y - TAG_OFFSET - _text_h(text, FS_SMALL) / 2),
        text,
        ha="center",
        color=PURPLE,
    )


def architecture(n_pages: int) -> Figure:
    """Три рівні: дані → обчислення → інтерфейс і звіти; окремо потік форми до моделей."""
    tiers = [
        ("Дані", DATA_NODES),
        ("Обчислення", COMPUTE_NODES),
        ("Інтерфейс і звіти", _ui_nodes(n_pages)),
    ]
    stack = max(len(n) * _node_h(n) + (len(n) - 1) * STACK_GAP for _, n in tiers)
    band_h = TITLE_H + stack + PAD
    route = _wrap(FORM_ROUTE, WIDTH - 2 * MARGIN, FS_SMALL)
    route_h = ROUTE_DROP + TAG_OFFSET + _text_h(route, FS_SMALL)
    total = band_h + route_h + 2 * MARGIN
    fig, ax = _canvas(total)
    top = total - MARGIN
    data, comp, ui = (
        _tier(ax, i, title, nodes, top, band_h) for i, (title, nodes) in enumerate(tiers)
    )
    _tier_links(ax, data, comp, ui)
    _form_route(ax, ui[-1], data[-1], top - band_h - ROUTE_DROP)
    return fig


# ---------- 3. Процедура моделювання ----------

TRAIN_W = 4.0
TEST_W = WIDTH - 2 * MARGIN - TRAIN_W - 0.35
FRAME_PAD = 0.15
HEAD_W = 4.6
EVAL_TEXT = "Оцінка на тесті: метрики, тест ДеЛонга, FINDRISC, калібрування"
EVAL_TAG = "не впливає на вибір моделі"
TEST_NOTE = (
    "Відкладена до навчання: ці дані не беруть участі в підборі параметрів, порогів і виборі моделі"
)
SPLIT_TEXT = "Стратифікований поділ: частка людей з ЦД2 однакова в обох частинах"
SPLIT_NOTE = "виконано до будь-якого навчання"


def _train_steps(cv_folds: int, best_name: str) -> list[Item]:
    return [
        (
            f"Крос-валідація на {cv_folds} фолдах: підбір параметрів кожної моделі за AUC",
            "імпутація медіаною і стандартизація — всередині кожного фолду",
        ),
        (
            "OOF-прогнози: кожну людину оцінює модель, яка її не бачила",
            "прогноз на фолді, не використаному в навчанні",
        ),
        (
            "Поріг Юдена для кожної моделі за OOF-прогнозами",
            "максимум суми чутливості й специфічності",
        ),
        (
            f"Вибір основної моделі за середнім AUC крос-валідації: {best_name}",
            "тестові дані у виборі не беруть участі",
        ),
        ("Перенавчання з підібраними параметрами на всій навчальній вибірці", ""),
    ]


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}{NBSP}%"


def _pipeline_head(ax: Axes, n_cohort: int, top: float, heights: tuple[float, float]) -> Box:
    """Аналітична вибірка → стратифікований поділ; повертає блок поділу."""
    cohort_h, split_h = heights
    text = _n_line("Аналітична вибірка", n_cohort)
    cohort = _box(ax, WIDTH / 2, top - cohort_h / 2, HEAD_W, text, look=DATA, bold=True, h=cohort_h)
    split_cy = cohort.bottom[1] - ARROW_GAP - split_h / 2
    split = _box(ax, WIDTH / 2, split_cy, HEAD_W, SPLIT_TEXT, note=SPLIT_NOTE, h=split_h)
    _arrow(ax, cohort.bottom, split.top)
    return split


def _train_frame(
    ax: Axes, title: str, steps: list[Item], hs: list[float], top: float, frame_h: float
) -> list[Box]:
    cx = MARGIN + TRAIN_W / 2
    _frame(ax, Box(cx, top - frame_h / 2, TRAIN_W, frame_h), TRAIN)
    _tag(ax, (cx, top - FRAME_TITLE_H / 2), title, ha="center", color=INK, bold=True)
    return _column(ax, steps, hs, cx, TRAIN_W - 2 * FRAME_PAD, top - FRAME_TITLE_H, look=DATA)


def _test_frame(ax: Axes, title: str, top: float, frame_h: float, refit: Box) -> Box:
    """Ізольована тестова частина: оцінка на рівні останнього кроку навчання."""
    cx, w = WIDTH - MARGIN - TEST_W / 2, TEST_W - 2 * FRAME_PAD
    _frame(ax, Box(cx, top - frame_h / 2, TEST_W, frame_h), TEST)
    _tag(ax, (cx, top - FRAME_TITLE_H / 2), title, ha="center", color=INK, bold=True)
    ev = _box(ax, cx, refit.cy, w, EVAL_TEXT, look=T2D, h=refit.h)
    tag_y = ev.top[1] + TAG_OFFSET + FS * LINE_EM * PT / 2
    _tag(ax, (cx, tag_y), EVAL_TAG, ha="center", color=ORANGE, bold=True)
    note = _wrap(TEST_NOTE, w, FS_SMALL)
    _tag(ax, (cx, (top - FRAME_TITLE_H + tag_y) / 2), note, ha="center")
    _arrow(ax, refit.right, ev.left, color=INK)
    return ev


def modelling_pipeline(
    n_cohort: int, n_train: int, n_test: int, cv_folds: int, best_name: str
) -> Figure:
    """Від вибірки до оцінки на тесті; тест візуально ізольований."""
    steps = _train_steps(cv_folds, best_name)
    hs = [_box_h(TRAIN_W - 2 * FRAME_PAD, t, n) for t, n in steps]
    hs[-1] = max(hs[-1], _box_h(TEST_W - 2 * FRAME_PAD, EVAL_TEXT))
    head = (
        _box_h(HEAD_W, _n_line("Аналітична вибірка", n_cohort), bold=True),
        _box_h(HEAD_W, SPLIT_TEXT, SPLIT_NOTE),
    )
    frame_h = FRAME_TITLE_H + sum(hs) + ARROW_GAP * (len(hs) - 1) + FRAME_PAD
    total = sum(head) + 3 * ARROW_GAP + frame_h + 2 * MARGIN
    fig, ax = _canvas(total)
    split = _pipeline_head(ax, n_cohort, total - MARGIN, head)
    top = split.bottom[1] - 2 * ARROW_GAP
    train_title = f"Навчальна вибірка\nN = {_n(n_train)} ({_pct(n_train, n_cohort)})"
    boxes = _train_frame(ax, train_title, steps, hs, top, frame_h)
    test_title = f"Тестова вибірка\nN = {_n(n_test)} ({_pct(n_test, n_cohort)})"
    _test_frame(ax, test_title, top, frame_h, boxes[-1])
    _split_routes(ax, split, top)
    return fig


def _split_routes(ax: Axes, split: Box, top: float) -> None:
    y = split.bottom[1] - ARROW_GAP
    train_x, test_x = MARGIN + TRAIN_W / 2, WIDTH - MARGIN - TEST_W / 2
    _route(ax, [split.bottom, (split.cx, y), (train_x, y), (train_x, top)])
    _route(ax, [(split.cx, y), (test_x, y), (test_x, top)], color=ORANGE, ls="--")


# ---------- 4. Поділ даних і різні N ----------

MAIN_W = 3.0
CHILD_W = 1.8
CHILD_GAP = 0.15
LEFT_CX = MARGIN + MAIN_W / 2
RIGHT_CX = WIDTH - MARGIN - (2 * CHILD_W + CHILD_GAP) / 2
COHORT_CX = 2.5
OR_W = 2.8
CHILD_SHIFT = (CHILD_W + CHILD_GAP) / 2
SPLIT_KEYS = ("flow", "train", "test", "complete", "logit", "frd_a", "frd_b", "frd_b_pos")
SPLIT_LEVELS: list[list[tuple[str, float, float, Look]]] = [
    [("cohort", COHORT_CX, MAIN_W, DATA), ("or", WIDTH - MARGIN - OR_W / 2, OR_W, SIDE)],
    [("train", LEFT_CX, MAIN_W, DATA), ("test", RIGHT_CX, MAIN_W, T2D)],
    [
        ("logit", LEFT_CX, MAIN_W, SIDE),
        ("a", RIGHT_CX - CHILD_SHIFT, CHILD_W, SIDE),
        ("b", RIGHT_CX + CHILD_SHIFT, CHILD_W, SIDE),
    ],
]  # рівні дерева: (ключ, центр x, ширина, вигляд)
SPLIT_GAP = 0.5


def _split_texts(n: dict[str, int | None]) -> dict[str, Item]:
    pos = n["frd_b_pos"]
    frd_b = _n_line("FINDRISC (b)", n["frd_b"]) + ("" if pos is None else f"\nз них з ЦД2: {pos}")
    return {
        "cohort": (_n_line("Аналітична вибірка", n["flow"]), ""),
        "or": (
            _n_line("Повні випадки для OR (табл. 3.5)", n["complete"]),
            "уся вибірка без пропусків в ознаках",
        ),
        "train": (
            _n_line("Навчальна вибірка", n["train"]),
            "крос-валідація, вибір моделі й порогу",
        ),
        "test": (_n_line("Тестова вибірка", n["test"]), "лише фінальна оцінка"),
        "logit": (
            _n_line("Логіт без регуляризації (табл. 4.1)", n["logit"]),
            "повні випадки навчальної вибірки",
        ),
        "a": (_n_line("FINDRISC (a)", n["frd_a"]), "тестові з повною анкетою"),
        "b": (frd_b, "недіагностовані з тестової вибірки"),
    }


def _split_level(
    ax: Axes, top: float, specs: list[tuple[str, float, float, Look]], texts: dict[str, Item]
) -> dict[str, Box]:
    """Ряд дерева: блоки однакової висоти, верх на рівні top."""
    h = max(_box_h(w, *texts[key]) for key, _, w, _ in specs)
    return {
        key: _box(ax, cx, top - h / 2, w, texts[key][0], note=texts[key][1], look=look, h=h)
        for key, cx, w, look in specs
    }


def _fork(ax: Axes, parent: Box, children: Sequence[Box], **style: Unpack[LineStyle]) -> None:
    """Стрілки від батька до дітей через спільну горизонталь."""
    y = (parent.bottom[1] + max(c.top[1] for c in children)) / 2
    for c in children:
        _route(ax, [parent.bottom, (parent.cx, y), (c.cx, y), c.top], **style)


def data_split(
    flow_n: int,
    n_train: int,
    n_test: int,
    n_complete_cases: int | None,
    n_logit_fit: int | None,
    n_frd_a: int | None,
    n_frd_b: int | None,
    n_frd_b_pos: int | None,
) -> Figure:
    """Звідки різні N: поділ train/test і підвибірки окремих аналізів. None — число не малюємо."""
    values = (flow_n, n_train, n_test, n_complete_cases, n_logit_fit, n_frd_a, n_frd_b, n_frd_b_pos)
    texts = _split_texts(dict(zip(SPLIT_KEYS, values, strict=True)))
    heights = [max(_box_h(w, *texts[k]) for k, _, w, _ in lv) for lv in SPLIT_LEVELS]
    total = sum(heights) + SPLIT_GAP * (len(SPLIT_LEVELS) - 1) + 2 * MARGIN
    fig, ax = _canvas(total)
    boxes: dict[str, Box] = {}
    top = total - MARGIN
    for level, h in zip(SPLIT_LEVELS, heights, strict=True):
        boxes |= _split_level(ax, top, level, texts)
        top -= h + SPLIT_GAP
    _split_arrows(ax, boxes)
    return fig


def _split_arrows(ax: Axes, b: dict[str, Box]) -> None:
    _arrow(ax, b["cohort"].right, b["or"].left, ls="--")
    _fork(ax, b["cohort"], [b["train"], b["test"]])
    _arrow(ax, b["train"].bottom, b["logit"].top, ls="--")
    _fork(ax, b["test"], [b["a"], b["b"]], ls="--")


# ---------- 5. Цільова змінна ----------

SPINE_W = 3.0
LEAF_W = 3.4
SPINE_CX = MARGIN + SPINE_W / 2
LEAF_CX = WIDTH - MARGIN - LEAF_W / 2
TREE_GAP = 0.45

type TreeRow = tuple[str, str, str, str, Look, str]


def _t1d_note(n: int | None) -> str:
    if n is None:
        return "ймовірний ЦД1 не відокремлюється"
    return f"ймовірний ЦД1 (діагноз до 30 років і інсулін зараз): N = {_n(n)}, не відокремлюється"


def _tree_rows(
    n_undefined: int, n_no_hba1c: int | None, n_t1d: int | None, thr: float
) -> list[TreeRow]:
    """(питання, підпис праворуч, лист, примітка листа, вигляд листа, підпис донизу)."""
    no_hba1c = _n_line("Виключено: немає HbA1c", n_no_hba1c)
    undefined = _n_line("Виключено: статус невизначений", n_undefined)
    refused = f"відмова, «не знаю» або немає відповіді, HbA1c < {thr:g} %"
    return [
        ("Є виміряний HbA1c?", "ні", no_hba1c, "на етапі відбору", EXCLUDED, "так"),
        ("Лікар казав про діабет?", "так", "ЦД2 (діагностований)", _t1d_note(n_t1d), T2D,
         "ні, межовий стан або немає відповіді"),
        (f"HbA1c ≥ {thr:g} %?", "так", "ЦД2 (недіагностований)", "", T2D, "ні"),
        ("Відповідь була «ні» або «межовий стан»?", "ні", undefined, refused, EXCLUDED, "так"),
    ]  # fmt: skip


def _tree_row(ax: Axes, row: TreeRow, cy: float, h: float) -> Box:
    question, right, leaf, note, look, _ = row
    q = _box(ax, SPINE_CX, cy, SPINE_W, question, look=QUESTION, bold=True, h=h)
    lf = _box(ax, LEAF_CX, cy, LEAF_W, leaf, note=note, look=look, h=h)
    _arrow(ax, q.right, lf.left)
    _tag(ax, ((q.right[0] + lf.left[0]) / 2, cy + 2 * TAG_OFFSET), right, ha="center")
    return q


def target_tree(
    n_excluded_undefined: int,
    n_excluded_no_hba1c: int | None = None,
    n_t1d_proxy: int | None = None,
    hba1c_threshold: float = HBA1C_THRESHOLD,
) -> Figure:
    """Дерево визначення ЦД2: відповідь про діагноз × HbA1c → ЦД2 / без ЦД2 / виключено.

    n_t1d_proxy — скільки серед діагностованих імовірних ЦД1 (None — без числа).
    """
    rows = _tree_rows(n_excluded_undefined, n_excluded_no_hba1c, n_t1d_proxy, hba1c_threshold)
    heights = [max(_box_h(SPINE_W, r[0], bold=True), _box_h(LEAF_W, r[2], r[3])) for r in rows]
    final_h = _box_h(SPINE_W, "Без ЦД2")
    total = sum(heights) + TREE_GAP * len(rows) + final_h + 2 * MARGIN
    fig, ax = _canvas(total)
    y, questions = total - MARGIN, []
    for row, h in zip(rows, heights, strict=True):
        questions.append(_tree_row(ax, row, y - h / 2, h))
        y -= h + TREE_GAP
    last = _box(ax, SPINE_CX, y - final_h / 2, SPINE_W, "Без ЦД2", look=NO_T2D, h=final_h)
    ends = [q.top for q in questions[1:]] + [last.top]
    for parent, end, row in zip(questions, ends, rows, strict=True):
        _down(ax, parent, end, row[5])
    return fig


def _down(ax: Axes, parent: Box, end: Point, label: str) -> None:
    _arrow(ax, parent.bottom, end)
    _tag(ax, (parent.cx + TAG_OFFSET, (parent.bottom[1] + end[1]) / 2), label)


# ---------- 6. FINDRISC: оригінал, модифікація, варіанти ----------

PANEL_W = (WIDTH - 2 * MARGIN - 0.2) / 2
ROW_H = FS * LINE_EM * PT + 0.05
VARIANT_COLS = (
    ("Варіант", 0.7),
    ("Хто входить (тестова вибірка)", 1.35),
    ("Пункт про глюкозу", 1.3),
    ("Ціль", 1.25),
    ("N і AUC", 1.05),
    ("Висновок", 1.15),
)

type ItemRow = tuple[str, str, str]


def _item_rows(original: Sequence[Item], modified: Sequence[Item]) -> tuple[list, list]:
    """Рядки (назва, бали, колір): прибрані пункти сірим, змінені бали акцентом."""
    old, new = dict(original), dict(modified)
    left = [(k, v, INK) if k in new else (k, f"{v} — прибрано", MUTED) for k, v in original]
    right = [
        (k, v, INK) if old.get(k, v) == v else (k, f"{v} (було {old[k]})", PURPLE)
        for k, v in modified
    ]
    return left, right


def _panel_h(title: str, rows: list[ItemRow], footer: str) -> float:
    inner = PANEL_W - 2 * PAD
    h = _text_h(_wrap(title, inner, FS, True), FS) + len(rows) * ROW_H + 3 * PAD
    return h + (_text_h(_wrap(footer, inner, FS_SMALL), FS_SMALL) + PAD if footer else 0.0)


def _panel(
    ax: Axes, x0: float, top: float, h: float, title: str, rows: list[ItemRow], footer: str
) -> None:
    """Список пунктів шкали: назва ліворуч, бали праворуч, примітки внизу."""
    inner = PANEL_W - 2 * PAD
    _frame(ax, Box(x0 + PANEL_W / 2, top - h / 2, PANEL_W, h), BAND)
    head = _wrap(title, inner, FS, True)
    _tag(
        ax,
        (x0 + PANEL_W / 2, top - PAD - _text_h(head, FS) / 2),
        head,
        ha="center",
        color=INK,
        bold=True,
    )
    y = top - 2 * PAD - _text_h(head, FS) - ROW_H / 2
    for name, points, color in rows:
        ax.text(x0 + PAD, y, name, ha="left", va="center", fontsize=FS, color=color)
        ax.text(x0 + PANEL_W - PAD, y, points, ha="right", va="center", fontsize=FS, color=color)
        y -= ROW_H
    if footer:
        text = _wrap(footer, inner, FS_SMALL)
        _tag(ax, (x0 + PAD, y + ROW_H / 2 - PAD - _text_h(text, FS_SMALL) / 2), text)


def _variant_cells(v: dict) -> list[str]:
    stats = [f"N{NBSP}={NBSP}{_n(v['n'])}"] if v.get("n") is not None else []
    stats += [f"AUC{NBSP}={NBSP}{v['auc']:.3f}"] if v.get("auc") is not None else []
    status = "придатний для порівняння" if v["valid"] else "витік: лише демонстрація"
    return [f"({v['key']})", v["who"], v["glucose"], v["target"], "\n".join(stats) or "—", status]


def _strip(ax: Axes, cells: list[str], top: float, look: Look, bold: bool) -> float:
    """Рядок таблиці варіантів; повертає його висоту."""
    widths = [w for _, w in VARIANT_COLS]
    h = max(_box_h(w, c, bold=bold) for c, w in zip(cells, widths, strict=True))
    x = MARGIN
    for cell, w in zip(cells, widths, strict=True):
        _box(ax, x + w / 2, top - h / 2, w, cell, look=look, bold=bold, h=h)
        x += w
    return h


def _strips(ax: Axes, variants: list[dict], top: float) -> None:
    top -= _strip(ax, [name for name, _ in VARIANT_COLS], top, HEADER, True)
    for v in variants:
        top -= _strip(ax, _variant_cells(v), top, VALID if v["valid"] else LEAK, False)


def _strips_h(variants: list[dict]) -> float:
    widths = [w for _, w in VARIANT_COLS]
    rows = [[n for n, _ in VARIANT_COLS]] + [_variant_cells(v) for v in variants]
    return sum(
        max(_box_h(w, c, bold=i == 0) for c, w in zip(r, widths, strict=True))
        for i, r in enumerate(rows)
    )


def findrisc_variants(
    items_original: Sequence[Item],
    items_modified: Sequence[Item],
    max_original: int,
    max_modified: tuple[int, int],
    variants: list[dict],
    notes: Sequence[str] = (),
) -> Figure:
    """Оригінал і модифікація шкали поруч; знизу три варіанти (a/b/c), витік — червоним.

    items_*: (пункт, бали «0/2/3/4»); max_modified: (з пунктом про глюкозу, без нього);
    variants: key, who, glucose, target, valid, n і auc (необов'язкові); notes — примітки
    до модифікації.
    """
    left, right = _item_rows(items_original, items_modified)
    titles = (
        f"Оригінальна FINDRISC: {len(items_original)} пунктів, максимум {max_original} балів",
        f"Модифікація для NHANES: {len(items_modified)} пунктів, максимум {max_modified[0]} "
        f"балів ({max_modified[1]} без пункту про глюкозу)",
    )
    footer = "\n".join(notes)
    panel_h = max(_panel_h(titles[0], left, ""), _panel_h(titles[1], right, footer))
    total = panel_h + ARROW_GAP + _strips_h(variants) + 2 * MARGIN
    fig, ax = _canvas(total)
    top = total - MARGIN
    _panel(ax, MARGIN, top, panel_h, titles[0], left, "")
    _panel(ax, WIDTH - MARGIN - PANEL_W, top, panel_h, titles[1], right, footer)
    _strips(ax, variants, top - panel_h - ARROW_GAP)
    return fig


# ---------- 7. Оцінка ризику людини ----------

PERSON_W = 4.35
SIDE_W = 2.1
SIDE_CX = WIDTH - MARGIN - SIDE_W / 2
GLUCOSE_TEXT = "Пункт анкети «підвищений цукор в анамнезі»"
FINDRISC_TEXT = "Бали FINDRISC і категорія ризику"
FINDRISC_NOTE = "вік, ІМТ, талія, активність, ліки від тиску, спадковість і пункт про цукор"
ONLY_FINDRISC = "лише у FINDRISC"


def _person_steps(model_names: Sequence[str], shap_model: str) -> list[Item]:
    return [
        ("Форма: вік, стать, зріст, вага, талія, тиск, холестерин, активність, анкета", ""),
        ("Перевірка значень", "допустимі діапазони; діастолічний тиск менший за систолічний"),
        ("ІМТ зі зросту і ваги", ""),
        ("«Не знаю» для тиску чи холестерину → медіана навчальної вибірки", ""),
        (
            "Моделі: " + ", ".join(m.replace(" ", NBSP) for m in model_names),
            "однаковий рядок ознак для всіх моделей",
        ),
        ("Ймовірність від кожної моделі і порівняння з її порогом", ""),
        ("SHAP: що найбільше вплинуло на результат", f"для моделі {shap_model}"),
        ("Сценарії «що якщо»: як зміниться ймовірність, якщо змінити ознаку", ""),
    ]


def _findrisc_branch(ax: Axes, form: Box, bmi: Box, probs: Box) -> None:
    """Окрема гілка: пункт про цукор і ІМТ ідуть у FINDRISC, а не в моделі."""
    h = _box_h(SIDE_W, GLUCOSE_TEXT)
    glucose = _box(ax, SIDE_CX, form.top[1] - h / 2, SIDE_W, GLUCOSE_TEXT, look=FORM, h=h)
    score = _box(ax, SIDE_CX, probs.cy, SIDE_W, FINDRISC_TEXT, note=FINDRISC_NOTE, look=FORM)
    _arrow(ax, form.right, (glucose.left[0], form.cy), color=PURPLE, lw=1.6)
    _arrow(ax, glucose.bottom, score.top, color=PURPLE, lw=1.6)
    x = SIDE_CX + TAG_OFFSET
    label = _wrap(ONLY_FINDRISC, WIDTH - MARGIN - x, FS_SMALL)
    _tag(ax, (x, (glucose.bottom[1] + score.top[1]) / 2), label, color=PURPLE)
    _arrow(ax, bmi.right, score.left, ls="--")


def person_flow(model_names: Sequence[str], shap_model: str) -> Figure:
    """Потік форми «Оцінка ризику»: від введення до SHAP і сценаріїв; FINDRISC окремо."""
    steps = _person_steps(model_names, shap_model)
    hs = [_box_h(PERSON_W, t, n) for t, n in steps]
    total = sum(hs) + ARROW_GAP * (len(hs) - 1) + 2 * MARGIN
    fig, ax = _canvas(total)
    boxes = _column(ax, steps, hs, MARGIN + PERSON_W / 2, PERSON_W, total - MARGIN, look=DATA)
    form, bmi, probs = boxes[0], boxes[2], boxes[5]
    _findrisc_branch(ax, form, bmi, probs)
    return fig


# ---------- 8. Попередня обробка ----------

STAGE_W = 1.8
PRE_GAP = 0.25


def _stages(ax: Axes, stages: Sequence[str], boxes: Sequence[Box]) -> None:
    """Рамки етапів ліворуч, кожна на висоту своїх кроків."""
    i = 0
    for k, (stage, group) in enumerate(groupby(stages)):
        n = len(list(group))
        top = boxes[i].top[1] + PRE_GAP / 2 - GAP
        bottom = boxes[i + n - 1].bottom[1] - PRE_GAP / 2 + GAP
        look = STAGE_LOOKS[k % len(STAGE_LOOKS)]
        _box(
            ax,
            MARGIN + STAGE_W / 2,
            (top + bottom) / 2,
            STAGE_W,
            stage,
            look=look,
            bold=True,
            h=top - bottom,
        )
        i += n


def preprocessing_flow(steps: Sequence[tuple[str, str, str]]) -> Figure:
    """Кроки обробки (крок, що робимо, етап) вертикально, згруповані за етапом."""
    step_w = WIDTH - 2 * MARGIN - STAGE_W - 0.2
    items = [(name, what) for name, what, _ in steps]
    hs = [_box_h(step_w, name, what, bold=True) for name, what in items]
    total = sum(hs) + PRE_GAP * (len(hs) - 1) + 2 * MARGIN
    fig, ax = _canvas(total)
    boxes = _column(
        ax, items, hs, WIDTH - MARGIN - step_w / 2, step_w, total - MARGIN, bold=True, gap=PRE_GAP
    )
    _stages(ax, [stage for *_, stage in steps], boxes)
    return fig


# ---------- 9. Оглядова карта проєкту ----------

MAP_HEADER = ("Етап", "Чим зроблено", "Де в коді", "Де це видно")
MAP_W = (1.35, 1.4, 2.4, 1.47)
MAP_GAP = 0.06
MAP_ROW_GAP = 0.1


def _map_heights(cells: Sequence[tuple[str, ...]]) -> list[float]:
    """Висота кожного рядка карти — за найвищою коміркою."""
    return [
        max(
            _box_h(w, text, bold=k == 0) for k, (w, text) in enumerate(zip(MAP_W, row, strict=True))
        )
        for row in cells
    ]


def _map_row(ax: Axes, row: Sequence[str], cy: float, h: float, *, header: bool = False) -> Box:
    """Один рядок карти: чотири комірки однакової висоти; повертає комірку «Етап»."""
    lefts = [MARGIN + sum(MAP_W[:k]) + k * MAP_GAP for k in range(len(MAP_W))]
    boxes = [
        _box(
            ax, left + w / 2, cy, w, text,
            look=HEADER if header else (DATA if k == 0 else PLAIN), bold=header or k == 0, h=h,
        )
        for k, (left, w, text) in enumerate(zip(lefts, MAP_W, row, strict=True))
    ]  # fmt: skip
    return boxes[0]


def project_map(rows: Sequence[tuple[str, str, str, str]]) -> Figure:
    """Карта проєкту: рядок на етап; колонки «Етап · Чим зроблено · Де в коді · Де це видно»."""
    cells = [MAP_HEADER, *rows]
    hs = _map_heights(cells)
    total = sum(hs) + MAP_ROW_GAP * (len(hs) - 1) + 2 * MARGIN
    fig, ax = _canvas(total)
    y, prev = total - MARGIN, None
    for i, (row, h) in enumerate(zip(cells, hs, strict=True)):
        stage = _map_row(ax, row, y - h / 2, h, header=i == 0)
        if prev is not None and i > 1:
            _arrow(ax, prev.bottom, stage.top)
        prev, y = stage, y - h - MAP_ROW_GAP
    return fig
