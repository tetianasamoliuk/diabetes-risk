"""Читання XPT та об'єднання модулів одного циклу."""

from pathlib import Path

import pandas as pd

from risk.config import MODULES, RAW_DIR
from risk.data.download import file_name

ZERO_EPS = 1e-70


def normalize_xpt_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """read_sas повертає SAS-нуль як ~5.4e-79; |x| < 1e-70 -> 0.0 у числових колонках."""
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].mask(out[num].abs() < ZERO_EPS, 0.0)
    return out


def read_xpt(path: Path) -> pd.DataFrame:
    """Читає XPT, нормалізує нулі, SEQN -> int64."""
    df = normalize_xpt_zeros(pd.read_sas(path, format="xport"))
    df["SEQN"] = df["SEQN"].astype("int64")
    return df


def load_cycle(cycle: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Усі модулі циклу, left-join на DEMO по SEQN; дублікати колонок праворуч відкидаються."""
    df = read_xpt(raw_dir / f"{file_name('DEMO', cycle)}.xpt")
    for module in MODULES:
        if module == "DEMO":
            continue
        part = read_xpt(raw_dir / f"{file_name(module, cycle)}.xpt")
        dup = part.columns.intersection(df.columns).drop("SEQN")
        df = df.merge(part.drop(columns=dup), on="SEQN", how="left", validate="1:1")
    return df.copy().assign(cycle=cycle)
