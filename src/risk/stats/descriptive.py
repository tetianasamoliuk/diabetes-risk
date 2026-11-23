"""Описова статистика за групами цільової змінної."""

import pandas as pd

STAT_COLS = ["n", "mean", "sd", "median", "q1", "q3", "iqr", "min", "max"]


def _summary(x: pd.Series) -> pd.Series:
    """Статистики однієї вибірки; NaN ігноруються, n — кількість непропущених."""
    x = x.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return pd.Series(
        {
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(ddof=1),
            "median": x.median(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": x.min(),
            "max": x.max(),
        }
    )


def describe_by_group(df: pd.DataFrame, cols: list[str], group: str = "t2d") -> pd.DataFrame:
    """Таблиця з MultiIndex (variable, group) і колонками STAT_COLS."""
    rows = {(c, g): _summary(sub[c]) for c in cols for g, sub in df.groupby(group, sort=True)}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=STAT_COLS)
    out.index = pd.MultiIndex.from_tuples(out.index, names=["variable", "group"])
    out["n"] = out["n"].astype(int)
    return out
