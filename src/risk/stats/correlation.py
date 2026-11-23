"""Кореляції Пірсона / Спірмена між неперервними ознаками."""

from typing import Literal

import pandas as pd

type Method = Literal["pearson", "spearman"]


def corr_matrix(df: pd.DataFrame, cols: list[str], method: Method = "pearson") -> pd.DataFrame:
    """Квадратна кореляційна матриця (попарне видалення пропусків)."""
    return df[cols].corr(method=method)
