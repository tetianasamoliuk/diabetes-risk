"""Відношення шансів (OR): 2×2 (Wald), однофакторний і багатофакторний логіт.

Неперервні фактори подаються на клінічний крок (вік на 10 років, ІМТ на 5 кг/м² тощо), щоб OR
було видно на графіку і легко пояснити; бінарні — як є.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from risk.config import (
    ACTIVE_MIN_PER_WEEK,
    BINARY_FACTORS,
    FEATURE_LABELS_UK,
    OR_SCALE,
    OR_SCALE_LABELS_UK,
    WAIST_CUTS_CM,
    Z95,
)
from risk.errors import DataValidationError

ACTIVE_H_PER_WEEK = ACTIVE_MIN_PER_WEEK / 60
MIN_ROWS_PER_PARAM = 10  # багатофакторний логіт: щонайменше 10 повних випадків на параметр
# (колонка, поріг, підпис): експозиція — значення ≥ порогу, референтна група — нижче порогу
CLINICAL_CUTS = (
    ("age", 45, "Вік ≥ 45 років (проти < 45)"),
    ("bmi", 30, "ІМТ ≥ 30 кг/м² (проти < 30)"),
    ("sbp", 140, "САТ ≥ 140 мм рт. ст. (проти < 140)"),
)
WAIST_LABEL = (
    f"Талія ≥ {WAIST_CUTS_CM[True][1]} см (чол.) / ≥ {WAIST_CUTS_CM[False][1]} см (жін.) "
    "(проти нижчої)"
)
ACTIVITY_LABEL = f"Активність < {ACTIVE_H_PER_WEEK:g} год/тиждень (проти ≥ {ACTIVE_H_PER_WEEK:g})"
CHOL_GROUPS = (
    ("Усі з відомим статусом прийому ліків", None),
    ("Не приймають ліки від холестерину", 0),
    ("Приймають ліки від холестерину", 1),
)


@dataclass(frozen=True, slots=True)
class OddsRatio:
    """OR з 95 % CI; n_exposed — кількість людей з фактором (для 2×2), scale — крок OR."""

    variable: str
    odds_ratio: float
    ci_low: float
    ci_high: float
    p: float
    n: int
    n_exposed: int = 0
    scale: float = 1.0
    label: str = ""


def or_from_2x2(a: float, b: float, c: float, d: float, variable: str = "") -> OddsRatio:
    """OR за таблицею 2×2 (a,b — фактор=1 з подією/без; c,d — фактор=0).

    OR = (a·d)/(b·c); log OR ± z₀.₉₇₅·sqrt(1/a+1/b+1/c+1/d); p — z-тест log OR / SE.
    Якщо хоч одна клітинка 0 — поправка Холдейна: +0.5 до всіх клітинок.
    n — сума вихідних (без поправки) клітинок.
    """
    n = int(a + b + c + d)
    a_raw, b_raw = a, b
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    p = 2 * stats.norm.sf(abs(log_or / se))
    return OddsRatio(
        variable,
        float(np.exp(log_or)),
        float(np.exp(log_or - Z95 * se)),
        float(np.exp(log_or + Z95 * se)),
        float(p),
        n,
        int(a_raw + b_raw),
    )


def _or_binary(x: pd.Series, y: pd.Series, name: str) -> OddsRatio:
    f1, f0 = x == 1, x == 0
    return or_from_2x2(
        int((f1 & (y == 1)).sum()),
        int((f1 & (y == 0)).sum()),
        int((f0 & (y == 1)).sum()),
        int((f0 & (y == 0)).sum()),
        name,
    )


def _or_continuous(x: pd.Series, y: pd.Series, name: str) -> OddsRatio:
    """OR на клінічний крок OR_SCALE: exp(coef·крок); CI на log-шкалі ± z·se·крок."""
    fit = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    scale = float(OR_SCALE.get(name, 1))
    coef, se = fit.params[name], fit.bse[name]
    return OddsRatio(
        name,
        float(np.exp(coef * scale)),
        float(np.exp((coef - Z95 * se) * scale)),
        float(np.exp((coef + Z95 * se) * scale)),
        float(fit.pvalues[name]),
        len(y),
        scale=scale,
    )


def _label(factor: str, continuous: bool) -> str:
    name = FEATURE_LABELS_UK.get(factor, factor)
    scale = OR_SCALE_LABELS_UK.get(factor)
    return f"{name.split(',')[0]}, {scale}" if continuous and scale else name


def univariate_or(df: pd.DataFrame, factor: str, outcome: str = "t2d") -> OddsRatio:
    """Бінарний фактор (0/1) → 2×2; неперервний → логіт, OR на клінічний крок OR_SCALE."""
    d = df[[factor, outcome]].dropna()
    x, y = d[factor], d[outcome]
    if factor in BINARY_FACTORS or x.nunique() <= 2:
        return replace(_or_binary(x, y, factor), label=_label(factor, False))
    return replace(_or_continuous(x, y, factor), label=_label(factor, True))


def or_for_threshold(
    df: pd.DataFrame, factor: str, threshold: float, outcome: str = "t2d"
) -> OddsRatio:
    """OR для factor ≥ threshold проти factor < threshold (наприклад, ІМТ ≥ 30 проти < 30)."""
    d = df[[factor, outcome]].dropna()
    return _or_binary((d[factor] >= threshold).astype(int), d[outcome], factor)


def _waist_or(df: pd.DataFrame, outcome: str) -> OddsRatio:
    """Талія ≥ верхньої межі WAIST_CUTS_CM для своєї статі проти нижчої."""
    d = df[["waist", "male", outcome]].dropna()
    cut = np.where(d["male"] == 1, WAIST_CUTS_CM[True][1], WAIST_CUTS_CM[False][1])
    return _or_binary((d["waist"] >= cut).astype(int), d[outcome], "waist")


def _low_activity_or(df: pd.DataFrame, outcome: str) -> OddsRatio:
    """Експозиція — активність нижче ACTIVE_H_PER_WEEK (інвертований прапорець)."""
    d = df[["pa_h_week", outcome]].dropna()
    return _or_binary((d["pa_h_week"] < ACTIVE_H_PER_WEEK).astype(int), d[outcome], "pa_h_week")


def _clinical_candidates(df: pd.DataFrame, outcome: str) -> list[tuple]:
    """(підпис, потрібні колонки, обчислення) для кожного клінічного порогу."""
    rows: list[tuple[str, tuple[str, ...], Callable[[], OddsRatio]]] = [
        (label, (col,), partial(or_for_threshold, df, col, cut, outcome))
        for col, cut, label in CLINICAL_CUTS
    ]
    rows.append((WAIST_LABEL, ("waist", "male"), partial(_waist_or, df, outcome)))
    rows.append((ACTIVITY_LABEL, ("pa_h_week",), partial(_low_activity_or, df, outcome)))
    return rows


def clinical_or_table(df: pd.DataFrame, outcome: str = "t2d") -> pd.DataFrame:
    """OR за клінічними порогами; пропускає пороги, для яких у даних немає колонок."""
    rows = []
    for label, cols, calc in _clinical_candidates(df, outcome):
        if all(c in df and df[c].notna().any() for c in cols):
            r = calc()
            rows.append((label, r.odds_ratio, r.ci_low, r.ci_high, r.p, r.n_exposed, r.n))
    cols = ["label", "odds_ratio", "ci_low", "ci_high", "p", "n_exposed", "n"]
    return pd.DataFrame(rows, columns=cols)


def cholesterol_by_medication(df: pd.DataFrame, outcome: str = "t2d") -> pd.DataFrame:
    """OR загального холестерину на 10 мг/дл (логіт): усі, без ліків і з ліками від холестерину.

    Перевіряє, чи обернений зв'язок холестерину з ЦД2 пояснюється лікуванням (статинами).
    """
    d = df[["tc", "chol_med", outcome]].dropna()
    rows = []
    for label, med in CHOL_GROUPS:
        g = d if med is None else d[d["chol_med"] == med]
        r = _or_continuous(g["tc"], g[outcome], "tc")
        rows.append((label, r.n, int(g[outcome].sum()), r.odds_ratio, r.ci_low, r.ci_high, r.p))
    cols = ["group", "n", "n_pos", "odds_ratio", "ci_low", "ci_high", "p"]
    return pd.DataFrame(rows, columns=cols)


def share_by_outcome(df: pd.DataFrame, col: str, outcome: str = "t2d") -> tuple[float, float]:
    """Частка (%) col == 1 серед outcome == 1 і серед outcome == 0 (рядки з відомим col)."""
    d = df[[col, outcome]].dropna()
    pct = d.groupby(outcome)[col].mean() * 100
    return float(pct.get(1, np.nan)), float(pct.get(0, np.nan))


def _vif(X: pd.DataFrame) -> pd.Series:
    vals = [
        np.nan if c == "const" else variance_inflation_factor(X.to_numpy(), i)
        for i, c in enumerate(X.columns)
    ]
    return pd.Series(vals, index=X.columns)


def _complete_cases(df: pd.DataFrame, features: list[str], outcome: str) -> pd.DataFrame:
    """Повні випадки; DataValidationError, якщо їх менше 10 на параметр моделі."""
    d = df[[*features, outcome]].dropna()
    if len(d) < MIN_ROWS_PER_PARAM * (len(features) + 1):
        raise DataValidationError(
            f"Замало повних випадків для багатофакторної моделі (N = {len(d)})"
        )
    return d


def multivariable_logit(
    df: pd.DataFrame, features: list[str], outcome: str = "t2d"
) -> pd.DataFrame:
    """Logit на повних випадках; індекс const + features, OR на одиницю ознаки."""
    d = _complete_cases(df, features, outcome)
    X = sm.add_constant(d[features].astype(float), has_constant="add")
    fit = sm.Logit(d[outcome], X).fit(disp=0)
    ci = np.exp(fit.conf_int())
    out = pd.DataFrame(
        {
            "coef": fit.params, "se": fit.bse, "odds_ratio": np.exp(fit.params),
            "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues, "vif": _vif(X), "sd": X.std(),
        }
    )  # fmt: skip
    out.loc["const", "sd"] = np.nan
    out["std_effect"] = out["coef"].abs() * out["sd"]
    out.attrs = {
        "n": len(d),
        "n_pos": int(d[outcome].sum()),
        "converged": bool(fit.mle_retvals["converged"]),
        "llf": float(fit.llf),
        "pseudo_r2": float(fit.prsquared),
    }
    return out


def per_clinical_step(or_table: pd.DataFrame) -> pd.DataFrame:
    """OR багатофакторної моделі на клінічний крок OR_SCALE (як в однофакторних) з підписом.

    OR = exp(β·крок), межі CI так само підносяться до степеня «крок»; бінарні ознаки як є.
    """
    t = or_table.drop(index="const", errors="ignore").copy()
    step = pd.Series([float(OR_SCALE.get(f, 1)) for f in t.index], index=t.index)
    for col in ("odds_ratio", "ci_low", "ci_high"):
        t[col] = t[col] ** step
    t["label"] = [_label(f, f in OR_SCALE) for f in t.index]
    return t


def rank_factors(or_table: pd.DataFrame, exclude: Sequence[str] = ("cycle_2017",)) -> pd.DataFrame:
    """Без const і exclude; за стандартизованим ефектом |β|·SD спаданням; rank з 1.

    Для бінарних ознак SD ≤ 0.5, тому їхній стандартизований ефект відносно занижений.
    """
    t = or_table.drop(index=["const", *exclude], errors="ignore")
    t = t.sort_values("std_effect", ascending=False).copy()
    t["rank"] = np.arange(1, len(t) + 1)
    return t
