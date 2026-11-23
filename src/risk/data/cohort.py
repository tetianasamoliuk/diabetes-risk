"""Відбір аналітичної вибірки з кроками блок-схеми і довідка про виключених."""

from dataclasses import dataclass

import pandas as pd

from risk.config import ADULT_AGE

LABELS = (
    "Усі учасники двох циклів",
    "Дорослі ≥ 20 років",
    "Без вагітних",
    "З виміряним HbA1c",
    "З визначеною цільовою змінною (аналітична вибірка)",
)


@dataclass(frozen=True, slots=True)
class FlowStep:
    """Крок блок-схеми: підпис, скільки лишилось і скільки відсіяно."""

    label: str
    n: int
    excluded: int


def filter_adults(df: pd.DataFrame) -> pd.DataFrame:
    """Лишає дорослих: вік не менший за ADULT_AGE (20 років)."""
    return df[df["age"] >= ADULT_AGE]


def filter_not_pregnant(df: pd.DataFrame) -> pd.DataFrame:
    """Виключає вагітних (колонка pregnant == True)."""
    return df[~df["pregnant"]]


def filter_hba1c_measured(df: pd.DataFrame) -> pd.DataFrame:
    """Лишає учасників із виміряним HbA1c (потрібен для визначення ЦД2)."""
    return df[df["hba1c"].notna()]


def filter_target_defined(df: pd.DataFrame) -> pd.DataFrame:
    """Лишає учасників із визначеною цільовою змінною t2d (не NaN)."""
    return df[df["t2d"].notna()]


def build_cohort(harmonized: pd.DataFrame) -> tuple[pd.DataFrame, list[FlowStep]]:
    """Послідовно застосовує фільтри; повертає вибірку і кроки (n, скільки відсіяно)."""
    filters = (filter_adults, filter_not_pregnant, filter_hba1c_measured, filter_target_defined)
    df = harmonized
    steps = [FlowStep(LABELS[0], len(df), 0)]
    for label, func in zip(LABELS[1:], filters, strict=True):
        after = func(df)
        steps.append(FlowStep(label, len(after), len(df) - len(after)))
        df = after
    return df.reset_index(drop=True), steps


def cohort_notes(harmonized: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, int | float]:
    """Хто виключений через відсутній HbA1c і скільки серед діагностованих ймовірного ЦД1.

    excl_no_mec — не проходили огляд у мобільному центрі (MEC), excl_examined_no_hba1c —
    оглянуті, але без аналізу, excl_hba1c_diagnosed — серед виключених без HbA1c мали діагноз.
    """
    no_hba1c = filter_not_pregnant(filter_adults(harmonized))
    no_hba1c = no_hba1c[no_hba1c["hba1c"].isna()]
    diagnosed = int((cohort["diq010"] == 1).sum())
    t1d = int((cohort["t1d_proxy"] == 1).sum())
    return {
        "excl_no_mec": int((~no_hba1c["examined"]).sum()),
        "excl_examined_no_hba1c": int(no_hba1c["examined"].sum()),
        "excl_hba1c_diagnosed": int((no_hba1c["diq010"] == 1).sum()),
        "diagnosed_n": diagnosed,
        "t1d_proxy_n": t1d,
        "t1d_proxy_pct": 100 * t1d / diagnosed if diagnosed else 0.0,
    }
