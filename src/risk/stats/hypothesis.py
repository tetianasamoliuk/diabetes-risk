"""Перевірка гіпотез: t-критерій Велча / Манна–Вітні, χ², поправка Холма, розмір ефекту.

Для неперервних ознак H0 стосується розподілу, а не середнього: Манна–Вітні порівнює ранги.
Розмір ефекту — AUC = U / (n₁·n₀), ймовірність, що значення в людини з ЦД2 вище, ніж без нього
(½ — ефекту немає), і рангово-бісеріальна кореляція r = 2·AUC − 1.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from risk.config import ALPHA, FEATURE_LABELS_UK

MIN_NORMALITY_N = 20  # на менших вибірках нормальності не довіряємо
# pct0 / pct1 — частка (%) зі значенням 1 у групах без ЦД2 і з ЦД2 (лише бінарні ознаки)
COLUMNS = [
    "variable", "label", "test", "statistic", "p", "p_holm", "reject_h0", "auc", "r_rb",
    "pct0", "pct1", "h0_text", "h1_text",
]  # fmt: skip


@dataclass(frozen=True, slots=True)
class GroupTestResult:
    """Результат порівняння двох груп; auc — розмір ефекту (NaN для χ²)."""

    variable: str
    test: str
    statistic: float
    p: float
    n0: int
    n1: int
    auc: float = float("nan")


def is_normal(x: pd.Series | np.ndarray, alpha: float = ALPHA) -> bool:
    """D'Agostino K²: нормальність не відхилена (p >= alpha). n < 20 → False."""
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < MIN_NORMALITY_N:
        return False
    return bool(stats.normaltest(arr).pvalue >= alpha)


def rank_auc(x0: np.ndarray, x1: np.ndarray) -> float:
    """P(X₁ > X₀) + ½·P(X₁ = X₀) = U₁ / (n₁·n₀): ймовірність вищого значення в групі 1."""
    u1 = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
    return float(u1 / (len(x0) * len(x1)))


def compare_continuous(
    x0: pd.Series | np.ndarray, x1: pd.Series | np.ndarray, variable: str = ""
) -> GroupTestResult:
    """Welch, якщо обидві групи нормальні, інакше Манн–Вітні (two-sided); плюс AUC."""
    a = np.asarray(x0, dtype=float)
    b = np.asarray(x1, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if is_normal(a) and is_normal(b):
        res, name = stats.ttest_ind(a, b, equal_var=False), "welch"
    else:
        res, name = stats.mannwhitneyu(a, b, alternative="two-sided"), "mann_whitney"
    stat, p = float(res.statistic), float(res.pvalue)
    return GroupTestResult(variable, name, stat, p, len(a), len(b), rank_auc(a, b))


def chi2_test(
    a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, variable: str = ""
) -> GroupTestResult:
    """χ² незалежності без поправки Єйтса; a — фактор, b — результат."""
    table = pd.crosstab(np.asarray(a), np.asarray(b))
    res = stats.chi2_contingency(table, correction=False)
    n_b = table.sum(axis=0).to_numpy()
    n0, n1 = (int(n_b[0]), int(n_b[1])) if len(n_b) > 1 else (int(n_b[0]), 0)
    return GroupTestResult(variable, "chi2", float(res.statistic), float(res.pvalue), n0, n1)


def holm(pvals: Sequence[float]) -> list[float]:
    """Скориговані p-value за Холмом (монотонні, ≤ 1); NaN лишаються NaN і не рахуються."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def _short_label(col: str) -> str:
    """Мітка без одиниць: частина назви до першої коми, регістр як у FEATURE_LABELS_UK."""
    return FEATURE_LABELS_UK.get(col, col).split(",")[0]


def _texts(label: str, binary: bool) -> tuple[str, str]:
    """H0/H1: для неперервних — про розподіл показника, для бінарних — про частку."""
    what, verb = (
        (f"частка «{label}»", "однакова")
        if binary
        else (f"розподіл показника «{label}»", "однаковий")
    )
    return (
        f"H0: {what} {verb} у групах з ЦД2 і без",
        f"H1: {what} відрізняється між групами з ЦД2 і без",
    )


def _shares(d: pd.DataFrame, col: str, outcome: str) -> list[float]:
    """Частка (%) зі значенням 1 бінарної ознаки в групах outcome = 0 і outcome = 1."""
    return [d.loc[d[outcome] == g, col].mean() * 100 for g in (0, 1)]


def _one(df: pd.DataFrame, col: str, outcome: str, continuous: bool) -> dict:
    d = df[[col, outcome]].dropna()
    if continuous:
        r = compare_continuous(d.loc[d[outcome] == 0, col], d.loc[d[outcome] == 1, col], col)
    else:
        r = chi2_test(d[col], d[outcome], col)
    h0, h1 = _texts(_short_label(col), not continuous)
    pct = _shares(d, col, outcome) if not continuous else [np.nan, np.nan]
    return {
        "variable": col,
        "label": FEATURE_LABELS_UK.get(col, col),
        "test": r.test,
        "statistic": r.statistic,
        "p": r.p,
        "auc": r.auc,
        "r_rb": 2 * r.auc - 1,
        "pct0": pct[0],
        "pct1": pct[1],
        "h0_text": h0,
        "h1_text": h1,
    }


def run_all_tests(
    df: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
    outcome: str = "t2d",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Усі тести + Holm; reject_h0 = p_holm < alpha. Без ознак — порожня таблиця COLUMNS."""
    rows = [_one(df, c, outcome, True) for c in continuous]
    rows += [_one(df, c, outcome, False) for c in categorical]
    out = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("p_holm", "reject_h0")])
    out["p_holm"] = holm(out["p"])
    out["reject_h0"] = (out["p_holm"] < alpha).astype(bool)
    return out[COLUMNS]
