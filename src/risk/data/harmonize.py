"""Гармонізація сирих змінних NHANES до єдиної схеми двох циклів."""

import numpy as np
import pandas as pd

from risk.config import CYCLES, DBP_COLS, HBA1C_THRESHOLD, PA_DOMAINS, SBP_COLS

MAX_DAYS = 7
MAX_MINUTES = 1440
T1D_MAX_AGE_DX = 30  # діагноз до цього віку і інсулін зараз — ознака ймовірного ЦД1
DX_UNDER_1_YEAR = 666  # код DID040 «менше 1 року»


def mean_bp(raw: pd.DataFrame, cols: list[str], drop_zero: bool) -> pd.Series:
    """Середнє доступних вимірів тиску; для ДАТ 0 («не вислухано») -> NaN."""
    part = raw[cols]
    if drop_zero:
        part = part.replace(0, np.nan)
    return part.mean(axis=1)


def yes_no(s: pd.Series) -> pd.Series:
    """1 -> 1.0, 2 -> 0.0, інше (відмова, не знаю, пропуск) -> NaN."""
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def pa_minutes_week(raw: pd.DataFrame) -> pd.Series:
    """GPAQ: сума днів x хвилин по 5 доменах; невалідний домен -> NaN всієї суми."""
    domains = []
    for did, days, mins in PA_DOMAINS:
        week = raw[days].where(raw[days] <= MAX_DAYS) * raw[mins].where(raw[mins] <= MAX_MINUTES)
        domains.append(week.where(raw[did] == 1, 0.0).where(raw[did].isin([1, 2])))
    return pd.concat(domains, axis=1).sum(axis=1, skipna=False)


def bp_medication(raw: pd.DataFrame) -> pd.Series:
    """BPQ050A (приймає зараз); пропуск анкети (немає гіпертензії / не призначали) -> 0."""
    med = yes_no(raw["BPQ050A"])
    skipped = (raw["BPQ020"] == 2) | (raw["BPQ040A"] == 2)
    return med.where(med.notna() | ~skipped, 0.0)


def cholesterol_medication(raw: pd.DataFrame) -> pd.Series:
    """BPQ100D (приймає зараз) з нулем для тих, кому питання не ставили.

    Ліки не призначали (BPQ090D == 2) або питання пропущено, бо про високий холестерин
    не казали (BPQ080 == 2) -> 0; інше -> NaN.
    """
    med = yes_no(raw["BPQ100D"])
    skipped = (raw["BPQ080"] == 2) | (raw["BPQ090D"] == 2)
    return med.where(med.notna() | ~skipped, 0.0)


def t1d_proxy(raw: pd.DataFrame) -> pd.Series:
    """Серед діагностованих (DIQ010 == 1): діагноз до 30 років і інсулін зараз -> 1, інакше 0."""
    age_dx = raw["DID040"].replace(DX_UNDER_1_YEAR, 0)
    proxy = (age_dx < T1D_MAX_AGE_DX) & (raw["DIQ050"] == 1)
    return proxy.astype(float).where(raw["DIQ010"] == 1)


def high_glucose_history(raw: pd.DataFrame) -> pd.Series:
    """DIQ160 == 1 або DIQ010 == 3 -> 1; DIQ160 == 2 -> 0; діагностованим (DIQ010 == 1) NaN."""
    diq160, diq010 = raw["DIQ160"], raw["DIQ010"]
    res = np.select([(diq160 == 1) | (diq010 == 3), diq160 == 2], [1.0, 0.0], np.nan)
    return pd.Series(res, index=raw.index).where(diq010 != 1)


def target(raw: pd.DataFrame) -> pd.Series:
    """ЦД2: 1 = DIQ010 == 1 або HbA1c >= 6.5; 0 = DIQ010 in {2,3} і HbA1c < 6.5; без HbA1c NaN."""
    hba1c, diq010 = raw["LBXGH"], raw["DIQ010"]
    pos = (diq010 == 1) | (hba1c >= HBA1C_THRESHOLD)
    neg = diq010.isin([2, 3]) & (hba1c < HBA1C_THRESHOLD)
    res = pd.Series(np.select([pos, neg], [1.0, 0.0], np.nan), index=raw.index)
    return res.where(hba1c.notna())


def target_undiagnosed(raw: pd.DataFrame) -> pd.Series:
    """Серед DIQ010 != 1 з виміряним HbA1c: HbA1c >= 6.5; інакше NaN."""
    hba1c = raw["LBXGH"]
    res = (hba1c >= HBA1C_THRESHOLD).astype(float)
    return res.where(hba1c.notna() & (raw["DIQ010"] != 1))


def _measurements(raw: pd.DataFrame, cycle: str) -> dict[str, pd.Series]:
    """Вік, стать, антропометрія, тиск, ліпіди і фізична активність."""
    pa = pa_minutes_week(raw)
    return {
        "age": raw["RIDAGEYR"],
        "male": (raw["RIAGENDR"] == 1).astype(float),
        "bmi": raw["BMXBMI"],
        "waist": raw["BMXWAIST"],
        "sbp": mean_bp(raw, SBP_COLS[cycle], drop_zero=False),
        "dbp": mean_bp(raw, DBP_COLS[cycle], drop_zero=True),
        "tc": raw["LBXTC"],
        "hdl": raw["LBDHDD"],
        "pa_min_week": pa,
        "pa_h_week": pa / 60,
    }


def _history(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Анкета: спадковість, ліки, глюкоза в анамнезі, ознака ЦД1, відповідь про діагноз."""
    return {
        "fam_dm": yes_no(raw["MCQ300C"]),
        "bp_med": bp_medication(raw),
        "high_glu_hist": high_glucose_history(raw),
        "chol_med": cholesterol_medication(raw),
        "t1d_proxy": t1d_proxy(raw),
        "diq010": raw["DIQ010"],
    }


def _outcomes(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """HbA1c і цільові змінні: ЦД2 загалом і недіагностований ЦД2."""
    return {"hba1c": raw["LBXGH"], "t2d": target(raw), "t2d_undiag": target_undiagnosed(raw)}


def _survey(raw: pd.DataFrame, cycle: str) -> dict[str, pd.Series]:
    """Ваги й дизайн опитування, ознаки обстеження і вагітності (для відбору)."""
    cfg = CYCLES[cycle]
    return {
        "wt_mec": raw[cfg["weight"]] * cfg["weight_k"],
        "psu": raw["SDMVPSU"],
        "strata": raw["SDMVSTRA"],
        "examined": raw["RIDSTATR"] == 2,
        "pregnant": raw["RIDEXPRG"] == 1,
    }


def harmonize(raw: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """Єдина схема колонок для обох циклів."""
    ids = {"SEQN": raw["SEQN"], "cycle": cycle, "cycle_2017": 1.0 if cycle == "2017-2020" else 0.0}
    parts = _measurements(raw, cycle) | _history(raw) | _outcomes(raw) | _survey(raw, cycle)
    return pd.DataFrame(ids | parts)
