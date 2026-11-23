"""Графіки проєкту (matplotlib). Кожна функція повертає Figure, нічого не показує."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullFormatter

from risk.config import FEATURE_LABELS_UK, FEATURES, MODEL_NAMES_UK
from risk.data.cohort import FlowStep
from risk.findrisc.score import reachable_categories
from risk.viz.style import (
    DIVERGING_CMAP,
    GROUP_LABELS,
    PALETTE,
    SEQUENTIAL_CMAP,
    apply_style,
)

if TYPE_CHECKING:  # shap важкий: імпортується лише у функціях SHAP-графіків
    import shap

_GROUP_COLORS = {0: str(PALETTE["t2d0"]), 1: str(PALETTE["t2d1"])}
LOG2_TICK_LIMIT = 6  # мітки осі OR не ширші за 1/64 … 64

# shap малює у спільну для процесу «поточну» фігуру pyplot, тож рисунки SHAP будуються по черзі.
# ponytail: один замок на процес — один рисунок SHAP за раз; якщо черга стане помітною,
# передавати shap власні осі (ax=…) там, де shap це вміє, і звузити або прибрати замок.
_SHAP_LOCK = threading.Lock()
_SHAP_SEED = 0  # розкид точок beeswarm однаковий у кожному збиранні


# ---------- спільні helper-и ----------


def _new(
    nrows: int = 1, ncols: int = 1, size: tuple[float, float] = (7, 4.5)
) -> tuple[Figure, np.ndarray]:
    """Фігура з constrained layout і єдиним стилем; осі — завжди 2D-масив (squeeze=False).

    Фігура поза реєстром pyplot: інший потік не отримає її як «поточну» через plt.gcf().
    """
    apply_style()
    fig = Figure(figsize=size, layout="constrained")
    return fig, fig.subplots(nrows, ncols, squeeze=False)


def _finish(fig: Figure, title: str) -> Figure:
    """Спільний вигляд: заголовок фігури."""
    fig.suptitle(title, fontweight="bold")
    return fig


def _label(name: str) -> str:
    """Українська мітка змінної (fallback — ім'я)."""
    return FEATURE_LABELS_UK.get(name, name)


def _nbsp(n: int) -> str:
    """Число з нерозривним пробілом як роздільником тисяч."""
    return f"{n:,}".replace(",", " ")


def _split(df: pd.DataFrame, col: str, group: str) -> dict[int, pd.Series]:
    """Значення колонки по групах (без пропусків)."""
    return {g: df.loc[df[group] == g, col].dropna() for g in (0, 1)}


# ---------- блок-схема ----------


def _flow_box(ax: Axes, xy: tuple[float, float], w: float, h: float, text: str, fc: str) -> None:
    """Прямокутник із центрованим текстом."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc=fc,
            ec=str(PALETTE["neutral"]),
            lw=1,
        )
    )
    ax.text(x, y, text, ha="center", va="center")


def flowchart(steps: Sequence[FlowStep]) -> Figure:
    """Вертикальна блок-схема відбору до аналітичної вибірки."""
    fig, axes = _new(size=(8, 1.3 * len(steps) + 0.5))
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps))
    for i, s in enumerate(steps):
        y = len(steps) - i - 0.5
        _flow_box(ax, (3.5, y), 5.5, 0.6, f"{s.label}\nN = {_nbsp(s.n)}", "#eaf1fb")
        if i > 0:
            ax.annotate("", (3.5, y + 0.3), (3.5, y + 0.7), arrowprops={"arrowstyle": "-|>"})
            _flow_excluded(ax, y + 0.5, s.excluded)
    return fig


def _flow_excluded(ax: Axes, y: float, excluded: int) -> None:
    """Бічний блок «Виключено» зі стрілкою від основної лінії."""
    ax.annotate("", (7.3, y), (3.5, y), arrowprops={"arrowstyle": "-|>", "color": "#6b6a66"})
    _flow_box(ax, (8.6, y), 2.6, 0.5, f"Виключено: {_nbsp(excluded)}", "#fdeee6")


# ---------- розподіли ----------


def hist_by_group(df: pd.DataFrame, col: str, label: str, group: str = "t2d") -> Figure:
    """Накладені гістограми щільності двох груп."""
    fig, axes = _new()
    ax = axes[0, 0]
    parts = _split(df, col, group)
    edges = np.histogram_bin_edges(pd.concat(parts.values()), bins=30)
    for g, vals in parts.items():
        ax.hist(
            vals,
            bins=edges,
            density=True,
            alpha=0.55,
            color=_GROUP_COLORS[g],
            label=GROUP_LABELS[g],
        )
    ax.set(xlabel=label, ylabel="Щільність")
    ax.legend()
    return _finish(fig, f"Розподіл: {label}")


def box_by_group(
    df: pd.DataFrame, cols: Sequence[str], labels: Sequence[str], group: str = "t2d"
) -> Figure:
    """Сітка діаграм розмаху (boxplot): по одному на змінну, дві групи поруч."""
    ncols = min(3, len(cols))
    nrows = -(-len(cols) // ncols)
    fig, axes = _new(nrows, ncols, size=(4 * ncols, 3.6 * nrows))
    for ax, col, lab in zip(axes.flat, cols, labels, strict=False):
        _box_one(ax, _split(df, col, group), lab)
    for ax in list(axes.flat)[len(cols) :]:
        ax.axis("off")
    return _finish(fig, "Порівняння груп")


def _box_one(ax: Axes, parts: dict[int, pd.Series], label: str) -> None:
    """Одна діаграма розмаху (boxplot) для двох груп."""
    bp = ax.boxplot(
        [parts[0], parts[1]], tick_labels=[GROUP_LABELS[0], GROUP_LABELS[1]], patch_artist=True,
        medianprops={"color": "black"},
    )  # fmt: skip
    for patch, g in zip(bp["boxes"], (0, 1), strict=True):
        patch.set(facecolor=_GROUP_COLORS[g], alpha=0.65)
    ax.set_ylabel(label)


def corr_heatmap(corr: pd.DataFrame, title: str) -> Figure:
    """Матриця кореляцій: дивергентна палітра −1…1, значення в клітинках."""
    n = len(corr)
    fig, axes = _new(size=(1.0 * n + 3, 0.9 * n + 2))
    ax = axes[0, 0]
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), [_label(c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(n), [_label(c) for c in corr.index])
    for i, j in np.ndindex(corr.shape):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Коефіцієнт кореляції")
    return _finish(fig, title)


# ---------- моделі ----------


def forest_plot(or_table: pd.DataFrame, title: str) -> Figure:
    """Форест-графік (forest plot) відношень шансів (log-шкала, лінія OR = 1).

    Підписи рядків — колонка label (з кроком, «Вік, на 10 років»), якщо вона є.
    """
    t = or_table.loc[or_table.index != "const"]
    labels = t["label"] if "label" in t else [_label(str(i)) for i in t.index]
    fig, axes = _new(size=(8, 0.5 * len(t) + 2))
    ax = axes[0, 0]
    y = np.arange(len(t))[::-1]
    xerr = [t["odds_ratio"] - t["ci_low"], t["ci_high"] - t["odds_ratio"]]
    ax.errorbar(t["odds_ratio"], y, xerr=xerr, fmt="o", color=str(PALETTE["t2d0"]), capsize=3, ms=6)
    ax.axvline(1, color=str(PALETTE["neutral"]), ls="--", lw=1)
    ax.set_xscale("log")
    _log2_ticks(ax, pd.concat([t["ci_low"], t["ci_high"], t["odds_ratio"]]))
    ax.set_yticks(y, list(labels))
    ax.set_xlabel("Відношення шансів (OR), логарифмічна шкала")
    return _finish(fig, title)


def _log2_ticks(ax: Axes, values: pd.Series) -> None:
    """Мітки осі OR степенями двійки (0.5, 1, 2, 4) звичайними числами, без 10⁰.

    Нескінченні й нульові межі CI (ідеальне розділення груп) ігноруються, а діапазон
    обмежено 2^±LOG2_TICK_LIMIT; без скінченних значень мітки лишаються стандартними.
    """
    ok = values[np.isfinite(values) & (values > 0)]
    if ok.empty:
        return
    lo = max(np.floor(np.log2(ok.min())), -LOG2_TICK_LIMIT)
    hi = min(np.ceil(np.log2(ok.max())), LOG2_TICK_LIMIT)
    ticks = 2.0 ** np.arange(min(lo, 0), max(hi, 0) + 1)
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> Figure:
    """ROC-криві моделей та діагональ випадкового класифікатора."""
    fig, axes = _new(size=(6, 5.5))
    ax = axes[0, 0]
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=_model_color(i),
            label=f"{MODEL_NAMES_UK.get(name, name)} (AUC = {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", color=str(PALETTE["neutral"]), label="Випадковий класифікатор")
    ax.set(xlabel="1 − специфічність", ylabel="Чутливість", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    return _finish(fig, "ROC-криві")


def _model_color(i: int) -> str:
    """Колір моделі за порядком (фіксований, без циклічного перебору понад 3)."""
    models = list(PALETTE["models"])
    return models[i % len(models)]


def calibration_curves(tables: dict[str, pd.DataFrame]) -> Figure:
    """Калібрувальні криві (mean_pred, frac_pos) та ідеальна діагональ."""
    fig, axes = _new(size=(6, 5.5))
    ax = axes[0, 0]
    for i, (name, t) in enumerate(tables.items()):
        ax.plot(
            t["mean_pred"],
            t["frac_pos"],
            marker="o",
            lw=2,
            color=_model_color(i),
            label=MODEL_NAMES_UK.get(name, name),
        )
    ax.plot([0, 1], [0, 1], ls="--", color=str(PALETTE["neutral"]), label="Ідеальне калібрування")
    ax.set(xlabel="Прогнозована ймовірність", ylabel="Фактична частка")
    ax.legend(loc="upper left")
    return _finish(fig, "Калібрування моделей")


def confusion_matrix_fig(tn: int, fp: int, fn: int, tp: int, title: str) -> Figure:
    """Матриця помилок 2×2 з числами."""
    fig, axes = _new(size=(5, 4.5))
    ax = axes[0, 0]
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap=SEQUENTIAL_CMAP)
    ax.grid(False)
    for i, j in np.ndindex(2, 2):
        dark = m[i, j] > m.max() / 2
        ax.text(
            j,
            i,
            _nbsp(int(m[i, j])),
            ha="center",
            va="center",
            fontsize=14,
            color="white" if dark else "black",
        )
    cls = ["без ЦД2", "з ЦД2"]
    ax.set_xticks([0, 1], [f"Прогноз: {c}" for c in cls])
    ax.set_yticks([0, 1], [f"Факт: {c}" for c in cls], rotation=90, va="center")
    return _finish(fig, title)


# ---------- FINDRISC ----------


def findrisc_distribution(score: pd.Series, y: pd.Series) -> Figure:
    """Гістограми балів FINDRISC по групах (дискретні біни 0..25)."""
    fig, axes = _new()
    ax = axes[0, 0]
    edges = np.arange(-0.5, 26.5, 1)
    for g in (0, 1):
        data = score[y == g].dropna()
        if data.empty:  # у користувацькому файлі група може бути порожньою
            continue
        ax.hist(
            data,
            bins=edges,
            density=True,
            alpha=0.55,
            color=_GROUP_COLORS[g],
            label=GROUP_LABELS[g],
        )
    ax.set(xlabel="Бали FINDRISC", ylabel="Частка учасників")
    ax.legend()
    return _finish(fig, "Розподіл балів FINDRISC")


def findrisc_categories(
    categories: pd.Series, y: pd.Series, max_score: int | None = None
) -> Figure:
    """Частка з ЦД2 у кожній категорії ризику; недосяжні при max_score категорії не показані."""
    order = reachable_categories(max_score)
    share = y.groupby(categories.astype(str)).mean().reindex(order) * 100
    n = categories.astype(str).value_counts().reindex(order).fillna(0).astype(int)
    fig, axes = _new(size=(8, 4.5))
    ax = axes[0, 0]
    bars = ax.bar(order, share.fillna(0), color=str(PALETTE["t2d1"]))
    ax.bar_label(
        bars,
        labels=[f"{v:.1f}%\nn = {_nbsp(c)}" for v, c in zip(share.fillna(0), n, strict=True)],
        fontsize=8,
    )
    ax.set(xlabel="Категорія ризику FINDRISC", ylabel="Частка з ЦД2, %")
    ax.margins(y=0.2)
    return _finish(fig, "Частка людей з ЦД2 за категоріями FINDRISC")


# ---------- поширеність ----------


def prevalence_bars(table: pd.DataFrame) -> Figure:
    """Парні стовпчики: незважена та зважена поширеність (CI на зважених)."""
    fig, axes = _new(size=(8, 4.5))
    ax = axes[0, 0]
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["unweighted_pct"], w, color=str(PALETTE["neutral"]), label="Незважена")
    yerr = [table["weighted_pct"] - table["ci_low"], table["ci_high"] - table["weighted_pct"]]
    ax.bar(
        x + w / 2,
        table["weighted_pct"],
        w,
        yerr=yerr,
        capsize=3,
        color=str(PALETTE["t2d0"]),
        label="Зважена (95% ДІ)",
    )
    ax.set_xticks(x, table["group"].astype(str))
    ax.set(ylabel="Поширеність ЦД2, %")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2
    )  # над осями: не накриє стовпчики
    return _finish(fig, "Поширеність ЦД2: незважена та зважена оцінки")


def prevalence_dots(table: pd.DataFrame) -> Figure:
    """Поширеність з різних джерел точками: label (два рядки), value, ci_low, ci_high.

    Відрізок довірчого інтервалу малюється лише там, де інтервал відомий.
    """
    fig, axes = _new(size=(9, 0.7 * len(table) + 1.6))
    ax = axes[0, 0]
    y = np.arange(len(table))[::-1]
    has_ci = (table["ci_low"].notna() & table["ci_high"].notna()).to_numpy()
    ci = table[has_ci]
    xerr = [ci["value"] - ci["ci_low"], ci["ci_high"] - ci["value"]]
    ax.errorbar(ci["value"], y[has_ci], xerr=xerr, fmt="none", ecolor=str(PALETTE["neutral"]))
    colors = [_country_color(str(s)) for s in table["label"]]
    ax.scatter(table["value"], y, s=50, color=colors, zorder=3)
    for v, yi in zip(table["value"], y, strict=True):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_yticks(y, table["label"].astype(str), fontsize=8.5)
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_xlim(0, float(np.nanmax([table["value"].max(), table["ci_high"].max()])) * 1.15)
    ax.set_xlabel("Поширеність діабету, %")
    return _finish(fig, "Поширеність діабету серед дорослих: США і Україна (різні методи)")


def _country_color(label: str) -> str:
    return str(PALETTE["t2d1"]) if label.startswith("Україна") else str(PALETTE["t2d0"])


# ---------- SHAP ----------


def _relabel(expl: shap.Explanation) -> shap.Explanation:
    """Explanation з українськими назвами ознак."""
    import shap

    names = [_label(n) for n in expl.feature_names]
    return shap.Explanation(
        values=expl.values, base_values=expl.base_values, data=expl.data, feature_names=names
    )


def _shap_attempt(draw: Callable[[], object]) -> Figure | None:
    """Одна спроба: нова поточна фігура, draw() малює в неї; None, якщо її підмінили.

    Streamlit після кожного запуску сторінки викликає plt.close("all") у своєму потоці, поза
    замком. Тоді shap домальовує в іншу, порожню фігуру або падає на ній. Фігуру знято
    з реєстру pyplot: наступне чуже close("all") готовий рисунок уже не зачепить.
    """
    apply_style()
    fig = plt.figure()
    try:
        draw()
    except Exception:  # будь-яка помилка shap на підміненій фігурі — привід малювати заново
        if plt.gcf() is fig:
            raise
    current = plt.gcf()
    plt.close(current)
    return fig if current is fig else None


def _shap_figure(draw: Callable[[], object]) -> Figure:
    """Рисунок shap поза реєстром pyplot; якщо фігуру підмінили, спроба повторюється."""
    with _SHAP_LOCK:
        # ponytail: повтор без ліміту — кожен викликаний завершенням чийогось запуску сторінки;
        # якщо повтори стануть помітні, передавати shap власні осі (ax=…) і прибрати pyplot
        while (fig := _shap_attempt(draw)) is None:
            pass
    return fig


def shap_beeswarm(expl: shap.Explanation) -> Figure:
    """SHAP-графік (beeswarm) з українськими підписами; усі ознаки без згортання."""
    import shap

    def draw() -> None:
        # shap 0.54 бере випадковість із глобального генератора numpy (параметра rng немає):
        # фіксуємо його на час виклику й повертаємо чужий стан
        state = np.random.get_state()
        np.random.set_state(np.random.RandomState(_SHAP_SEED).get_state())
        try:
            shap.plots.beeswarm(_relabel(expl), max_display=len(expl.feature_names), show=False)
        finally:
            np.random.set_state(state)

    fig = _shap_figure(draw)
    fig.axes[0].set_xlabel("Значення SHAP (вплив на прогноз, log-odds)")
    cbar = fig.axes[-1]  # колірна шкала: низьке/високе значення ознаки
    cbar.set_ylabel("Значення ознаки")
    cbar.set_yticklabels(["низьке", "високе"])
    return fig


def shap_bar(expl: shap.Explanation) -> Figure:
    """SHAP: середня абсолютна важливість ознак (усі ознаки, без «Sum of N other»).

    shap підписує стовпчики зі знаком («+0.87»), а середнє |SHAP| завжди невід'ємне — знак знято.
    """
    import shap

    fig = _shap_figure(
        lambda: shap.plots.bar(_relabel(expl), max_display=len(expl.feature_names), show=False)
    )
    ax = fig.axes[0]
    ax.set_xlabel("Середнє |SHAP| (внесок у log-odds)")
    for text in ax.texts:
        text.set_text(text.get_text().removeprefix("+"))
    return fig


def shap_waterfall(expl_row: shap.Explanation) -> Figure:
    """SHAP waterfall для одного спостереження."""
    import shap

    return _shap_figure(
        lambda: shap.plots.waterfall(_relabel(expl_row), max_display=len(FEATURES) + 1, show=False)
    )
