"""Тести пакета графіків: фігури повертаються, PNG валідний."""

import io
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from risk.findrisc.score import CATEGORY_NAMES
from risk.viz import figures as f
from risk.viz.style import GROUP_LABELS, PALETTE, apply_style, close, fig_to_png

RNG = np.random.default_rng(0)


@dataclass(frozen=True)
class Step:
    """Крок потоку відбору для перевірки `flowchart`."""

    label: str
    n: int
    excluded: int


@pytest.fixture(autouse=True)
def _close_all() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t2d": RNG.integers(0, 2, 200),
            "age": RNG.normal(50, 10, 200),
            "bmi": RNG.normal(28, 5, 200),
            "xx": RNG.normal(0, 1, 200),
        }
    )


def _ok(fig: Figure) -> None:
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_style_and_palette() -> None:
    apply_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["savefig.bbox"] == "tight"
    assert len(PALETTE["models"]) == 3
    assert GROUP_LABELS == {0: "Без ЦД2", 1: "З ЦД2"}


def test_png_dpi_and_close() -> None:
    fig, _ = plt.subplots()
    small, big = fig_to_png(fig, 100), fig_to_png(fig, 300)
    assert big.startswith(b"\x89PNG")
    assert Image.open(io.BytesIO(big)).size[0] > Image.open(io.BytesIO(small)).size[0]
    close(fig)
    assert not plt.get_fignums()


def test_flowchart() -> None:
    steps = [Step("Усі", 20000, 0), Step("Дорослі", 12345, 7655), Step("Когорта", 9000, 3345)]
    fig = f.flowchart(steps)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("N = 20 000" in t for t in texts)
    assert any(t.startswith("Виключено") for t in texts)
    assert not fig.axes[0].axison


def test_hist_and_box(df: pd.DataFrame) -> None:
    _ok(f.hist_by_group(df, "age", "Вік, років"))
    fig = f.box_by_group(df, ["age", "bmi", "xx", "age"], ["a", "b", "c", "d"])
    _ok(fig)


def test_corr_heatmap(df: pd.DataFrame) -> None:
    fig = f.corr_heatmap(df[["age", "bmi", "xx"]].corr(), "Кореляції")
    _ok(fig)
    assert len(fig.axes[0].texts) == 9


def test_forest_skips_const() -> None:
    t = pd.DataFrame(
        {"odds_ratio": [0.1, 1.5, 2.0], "ci_low": [0.05, 1.1, 1.4], "ci_high": [0.2, 2.0, 2.9]},
        index=["const", "age", "bmi"],
    )
    ax = f.forest_plot(t, "OR").axes[0]
    assert len(ax.get_yticklabels()) == 2
    assert ax.get_xscale() == "log"
    assert [x.get_text() for x in ax.get_xticklabels()] == ["1", "2", "4"]


def test_forest_with_infinite_or_zero_ci() -> None:
    # Ідеальне розділення груп дає CI 0 або inf: графік будується, мітки в межах 2^±6
    t = pd.DataFrame(
        {"odds_ratio": [3.0, 0.5], "ci_low": [1.5, 0.0], "ci_high": [np.inf, 0.9]},
        index=["age", "bmi"],
    )
    ax = f.forest_plot(t, "OR").axes[0]
    ticks = [float(x.get_text()) for x in ax.get_xticklabels()]
    assert ticks == [0.5, 1, 2, 4] and len(ax.get_yticklabels()) == 2
    none = t.assign(odds_ratio=np.nan, ci_low=0.0, ci_high=np.inf)
    _ok(f.forest_plot(none, "OR"))
    wide = t.assign(ci_low=[1e-9, 0.1], ci_high=[1e9, 2.0])
    wide_ticks = [float(x.get_text()) for x in f.forest_plot(wide, "OR").axes[0].get_xticklabels()]
    assert min(wide_ticks) == 2**-6 and max(wide_ticks) == 2**6


def test_forest_uses_label_column() -> None:
    t = pd.DataFrame(
        {"odds_ratio": [0.7, 1.5], "ci_low": [0.6, 1.1], "ci_high": [0.8, 2.0],
         "label": ["Вік, на 10 років", "Стать"]},
        index=["age", "male"],
    )  # fmt: skip
    ax = f.forest_plot(t, "OR").axes[0]
    assert [x.get_text() for x in ax.get_yticklabels()] == ["Вік, на 10 років", "Стать"]
    assert [x.get_text() for x in ax.get_xticklabels()] == ["0.5", "1", "2"]


def test_roc_and_calibration() -> None:
    x = np.linspace(0, 1, 20)
    fig = f.roc_curves({"logit": (x, x**0.5, 0.8), "rf": (x, x**0.4, 0.85)})
    assert len(fig.axes[0].lines) == 3
    cal = pd.DataFrame({"mean_pred": x, "frac_pos": x})
    _ok(f.calibration_curves({"logit": cal, "hgb": cal}))


def test_confusion_and_metrics() -> None:
    fig = f.confusion_matrix_fig(100, 20, 10, 30, "Матриця")
    assert len(fig.axes[0].texts) == 4


def test_findrisc() -> None:
    score = pd.Series(RNG.integers(0, 26, 300))
    y = pd.Series(RNG.integers(0, 2, 300))
    _ok(f.findrisc_distribution(score, y))
    cats = pd.Series(RNG.choice(CATEGORY_NAMES[:4], 300))
    fig = f.findrisc_categories(cats, y)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == CATEGORY_NAMES
    short = f.findrisc_categories(cats, y, max_score=20)
    assert [t.get_text() for t in short.axes[0].get_xticklabels()] == CATEGORY_NAMES[:4]


def test_prevalence_dots_ci_only_where_known() -> None:
    table = pd.DataFrame(
        {
            "label": ["США (NHANES)\nHbA1c, ≥ 20 років", "Україна (IDF 2024)\nмодельна оцінка"],
            "value": [13.9, 6.0],
            "ci_low": [13.0, np.nan],
            "ci_high": [14.8, np.nan],
        }
    )
    fig = f.prevalence_dots(table)
    ax = fig.axes[0]
    assert len(ax.collections) == 2  # один відрізок CI (LineCollection) і точки
    assert [t.get_text() for t in ax.texts] == ["13.9%", "6.0%"]
    assert "різні методи" in fig._suptitle.get_text()
    plt.close(fig)


def test_prevalence() -> None:
    t = pd.DataFrame(
        {
            "group": ["A", "B"],
            "unweighted_pct": [10, 20],
            "weighted_pct": [11, 19],
            "ci_low": [9, 17],
            "ci_high": [13, 21],
        }
    )
    _ok(f.prevalence_bars(t))


@pytest.mark.parametrize("pcts", [[18.2, 19.3], [30.0, 8.0, 99.0], [5.0, 5.0, 5.0, 5.0]])
def test_prevalence_bars_legend_clear_of_bars(pcts: list[float]) -> None:
    # легенда не накриває жодного стовпчика і мітки похибки за будь-яких значень
    pct = pd.Series(pcts)
    t = pd.DataFrame(
        {
            "group": [str(i) for i in range(len(pct))],
            "unweighted_pct": pct,
            "weighted_pct": pct * 0.7,
            "ci_low": pct * 0.6,
            "ci_high": pct * 0.8,
        }
    )
    fig = f.prevalence_bars(t)
    fig.canvas.draw()
    ax = fig.axes[0]
    box = ax.get_legend().get_window_extent()
    parts = [p.get_window_extent() for p in ax.patches] + [
        ln.get_window_extent() for ln in ax.lines
    ]
    assert parts
    assert not any(box.overlaps(p) for p in parts)


def test_shap() -> None:
    names = ["age", "bmi", "custom"]
    vals = RNG.normal(size=(50, 3))
    expl = shap.Explanation(
        values=vals, base_values=np.zeros(50), data=RNG.normal(size=(50, 3)), feature_names=names
    )
    for make in (f.shap_beeswarm, f.shap_bar):
        fig = make(expl)
        _ok(fig)
        text = " ".join(t.get_text() for t in fig.findobj(Text))
        for eng in ("Feature value", "SHAP value (impact", "mean(|SHAP", "other features"):
            assert eng not in text
        assert "Значення" in text or "Середнє" in text
    assert "Значення ознаки" in " ".join(t.get_text() for t in f.shap_beeswarm(expl).findobj(Text))
    row = shap.Explanation(values=vals[0], base_values=0.1, data=expl.data[0], feature_names=names)
    _ok(f.shap_waterfall(row))


def test_shap_beeswarm_reproducible_keeps_global_rng() -> None:
    # той самий розкид точок у кожному збиранні; чужий глобальний генератор не зачеплено
    expl = shap.Explanation(
        values=RNG.normal(size=(50, 3)), base_values=np.zeros(50), data=RNG.normal(size=(50, 3)),
        feature_names=["age", "bmi", "custom"],
    )  # fmt: skip
    before = np.random.get_state()
    first, second = (fig_to_png(f.shap_beeswarm(expl), 100) for _ in range(2))
    assert first == second
    np.testing.assert_equal(np.random.get_state(), before)


@pytest.mark.parametrize("plot", ["beeswarm", "bar", "waterfall"])
def test_shap_survives_close_all_from_other_thread(
    plot: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Streamlit після кожного запуску сторінки викликає plt.close("all") у своєму потоці;
    # тут це стається одразу після того, як shap домалював, ще до повернення рисунка
    names = ["age", "bmi", "custom"]
    vals, data = RNG.normal(size=(50, 3)), RNG.normal(size=(50, 3))
    expl = shap.Explanation(values=vals, base_values=np.zeros(50), data=data, feature_names=names)
    row = shap.Explanation(values=vals[0], base_values=0.1, data=data[0], feature_names=names)
    build = {
        "beeswarm": lambda: f.shap_beeswarm(expl),
        "bar": lambda: f.shap_bar(expl),
        "waterfall": lambda: f.shap_waterfall(row),
    }[plot]
    clean = fig_to_png(build(), 50)
    original, calls = getattr(shap.plots, plot), []

    def _draw_then_close_all(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        if not calls:
            plt.close("all")
        calls.append(plot)
        return result

    monkeypatch.setattr(shap.plots, plot, _draw_then_close_all)
    fig = build()
    assert not plt.get_fignums()  # готовий рисунок поза pyplot: чуже close("all") його не зачепить
    assert fig_to_png(fig, 50) == clean


def test_shap_error_after_close_all_is_retried_own_error_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # shap падає на підміненій фігурі (так буває з waterfall) — малюємо заново;
    # помилка на власній фігурі — справжня, її не ховаємо
    expl = shap.Explanation(
        values=RNG.normal(size=(30, 2)), base_values=np.zeros(30), data=RNG.normal(size=(30, 2)),
        feature_names=["age", "bmi"],
    )  # fmt: skip
    clean = fig_to_png(f.shap_bar(expl), 50)
    original, calls = shap.plots.bar, []

    def _close_all_then_fail(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            plt.close("all")
            raise IndexError("list index out of range")
        return original(*args, **kwargs)

    monkeypatch.setattr(shap.plots, "bar", _close_all_then_fail)
    assert fig_to_png(f.shap_bar(expl), 50) == clean
    assert len(calls) == 2

    def _fail(*args: object, **kwargs: object) -> object:
        raise ValueError("справжня помилка")

    monkeypatch.setattr(shap.plots, "bar", _fail)
    with pytest.raises(ValueError, match="справжня помилка"):
        f.shap_bar(expl)
    monkeypatch.setattr(shap.plots, "bar", original)
    assert f.shap_bar(expl).axes  # замок звільнено після помилки


def test_figures_stay_outside_pyplot(df: pd.DataFrame) -> None:
    # фігура поза реєстром pyplot не стає «поточною» для інших потоків застосунку
    x = np.linspace(0, 1, 5)
    odds = pd.DataFrame({"odds_ratio": [1.5], "ci_low": [1.1], "ci_high": [2.0]}, index=["age"])
    prev = pd.DataFrame(
        {"group": ["A"], "unweighted_pct": [10], "weighted_pct": [11],
         "ci_low": [9], "ci_high": [13]}
    )  # fmt: skip
    dots = pd.DataFrame({"label": ["США"], "value": [13.9], "ci_low": [13.0], "ci_high": [14.8]})
    score, y = pd.Series([3, 12, 20]), pd.Series([0, 1, 1])
    builders = [
        lambda: f.flowchart([Step("Усі", 10, 0), Step("Дорослі", 8, 2)]),
        lambda: f.hist_by_group(df, "age", "Вік"),
        lambda: f.box_by_group(df, ["age"], ["Вік"]),
        lambda: f.corr_heatmap(df[["age", "bmi"]].corr(), "Кореляції"),
        lambda: f.forest_plot(odds, "OR"),
        lambda: f.roc_curves({"logit": (x, x, 0.5)}),
        lambda: f.calibration_curves({"logit": pd.DataFrame({"mean_pred": x, "frac_pos": x})}),
        lambda: f.confusion_matrix_fig(1, 2, 3, 4, "Матриця"),
        lambda: f.findrisc_distribution(score, y),
        lambda: f.findrisc_categories(pd.Series(CATEGORY_NAMES[:3]), y),
        lambda: f.prevalence_bars(prev),
        lambda: f.prevalence_dots(dots),
    ]
    for build in builders:
        fig = build()
        assert not plt.get_fignums()
        dpi = Image.open(io.BytesIO(fig_to_png(fig))).info["dpi"]
        assert [round(d) for d in dpi] == [300, 300]
        close(fig)  # як у застосунку: для фігури поза pyplot нічого не робить


def test_shap_bar_values_without_plus_sign() -> None:
    # середнє |SHAP| невід'ємне, тож знак «+» біля чисел зайвий
    expl = shap.Explanation(
        values=RNG.normal(size=(30, 2)), base_values=np.zeros(30), data=RNG.normal(size=(30, 2)),
        feature_names=["age", "bmi"],
    )  # fmt: skip
    labels = [t.get_text() for t in f.shap_bar(expl).axes[0].texts]
    assert labels and not any(s.startswith("+") for s in labels)
    assert all(float(s) > 0 for s in labels)


def test_figures_module_does_not_import_shap() -> None:
    # shap важкий (сотні МБ): графіки без SHAP не мають його завантажувати
    code = "import sys, risk.viz.figures; sys.exit('shap' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_findrisc_distribution_empty_group() -> None:
    score = pd.Series(RNG.integers(0, 26, 50))
    _ok(f.findrisc_distribution(score, pd.Series(np.zeros(50, dtype=int))))
