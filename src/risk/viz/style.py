"""Єдиний стиль графіків: палітра, rcParams, експорт у PNG."""

import io

import matplotlib

matplotlib.use("Agg")  # без GUI: PNG у потоках Streamlit і в тестах на macOS

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Сині й помаранчеві відтінки розрізняють і люди з порушеннями кольорового зору;
# models — кольори трьох моделей, які попарно добре відрізняються.
PALETTE: dict[str, str | list[str]] = {
    "t2d0": "#2a78d6",
    "t2d1": "#eb6834",
    "neutral": "#6b6a66",
    "accent": "#6250d6",
    "models": ["#2a78d6", "#eb6834", "#1baf7a"],
}

GROUP_LABELS: dict[int, str] = {0: "Без ЦД2", 1: "З ЦД2"}

# Дивергентна карта: синій -> нейтральний сірий -> помаранчевий (без кольору в середині)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "risk_div", [str(PALETTE["t2d0"]), "#f2f2f0", str(PALETTE["t2d1"])]
)
# Послідовна карта (один відтінок) для матриці помилок
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("risk_seq", ["#f2f6fc", str(PALETTE["t2d0"])])


def apply_style() -> None:
    """Застосувати єдині rcParams (кирилиця, приглушена сітка, без верхньої/правої рамки)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.grid": True,
            "grid.color": "#d9d9d6",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
        }
    )


def fig_to_png(fig: Figure, dpi: int = 300, tight: bool = True) -> bytes:
    """Відрендерити фігуру у PNG-байти.

    tight=False — рівно по полотну фігури, без відступу `bbox tight` (схеми мають власне поле).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else fig.bbox_inches)
    return buf.getvalue()


def close(fig: Figure) -> None:
    """Звільнити пам'ять фігури."""
    plt.close(fig)
