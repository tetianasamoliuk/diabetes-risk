"""Порівняння ROC-AUC тестом DeLong, швидка реалізація Sun & Xu.

DeLong E.R., DeLong D.M., Clarke-Pearson D.L. (1988). Biometrics 44(3):837–845.
Sun X., Xu W. (2014). Fast Implementation of DeLong's Algorithm for Comparing the Areas
Under Correlated ROC Curves. IEEE Signal Processing Letters 21(11):1389–1393.

Позначення: X_i — скори m позитивних (y == 1), Y_j — скори n негативних;
ψ(X, Y) = 1, якщо X > Y; ½, якщо X = Y; 0 інакше.  AUC = (1/mn)·Σ_i Σ_j ψ(X_i, Y_j).
Структурні компоненти: V10(X_i) = (1/n)·Σ_j ψ(X_i, Y_j),  V01(Y_j) = (1/m)·Σ_i ψ(X_i, Y_j).
Коваріація AUC k моделей: S = S10/m + S01/n, де S10[r, s] = cov(V10_r, V10_s) з дільником m − 1,
S01 — так само по негативних з дільником n − 1.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata

from risk.config import ALPHA
from risk.errors import DataValidationError

MIN_PER_CLASS = 2  # дисперсії з дільником m − 1 / n − 1 потребують ≥ 2 спостережень у класі


@dataclass(frozen=True, slots=True)
class DelongResult:
    """Результат тесту DeLong для двох AUC на тих самих спостереженнях."""

    auc_a: float
    auc_b: float
    diff: float  # auc_a − auc_b
    se_diff: float
    z: float
    p: float  # двобічний
    ci_low: float  # CI для diff
    ci_high: float


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Середні ранги (з 1) уздовж останньої осі: зв'язаним значенням — середнє їхніх позицій."""
    return rankdata(x, method="average", axis=-1)


def _validate(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Маска позитивних (y == 1) і скори форми (k, n); кидає DataValidationError."""
    pos = np.asarray(y).ravel() == 1
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    if s.shape[1] != pos.size:
        raise DataValidationError(f"Довжина скорів {s.shape[1]} ≠ довжині y {pos.size}")
    if not np.isfinite(s).all():
        raise DataValidationError("Скори містять NaN або inf")
    if min(pos.sum(), pos.size - pos.sum()) < MIN_PER_CLASS:
        raise DataValidationError(f"Потрібно ≥ {MIN_PER_CLASS} позитивних і негативних")
    return pos, s


def fast_delong(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC (k,) і коваріаційна матриця DeLong (k, k) для скорів форми (k, n)."""
    pos, s = _validate(y, scores)
    m, n = int(pos.sum()), int((~pos).sum())
    # Sun & Xu: подвійна сума ψ через midrank за O(n log n).
    # Tz — midrank в об'єднаній вибірці, Tx / Ty — midrank усередині позитивних / негативних.
    # Tz_i − Tx_i = к-сть негативних нижче X_i + ½·зв'язки  ⇒  V10(X_i) = (Tz_i − Tx_i) / n.
    # Tz_j − Ty_j = к-сть позитивних нижче Y_j + ½·зв'язки ⇒  V01(Y_j) = 1 − (Tz_j − Ty_j) / m.
    tz = compute_midrank(s)
    v10 = (tz[:, pos] - compute_midrank(s[:, pos])) / n
    v01 = 1.0 - (tz[:, ~pos] - compute_midrank(s[:, ~pos])) / m
    aucs = v10.mean(axis=1)  # AUC = середнє V10 = середнє V01
    # np.cov: рядки — моделі, дільник m − 1 / n − 1 (незміщена оцінка, як у DeLong 1988)
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return aucs, cov


def auc_ci(y: np.ndarray, score: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """AUC і (1 − alpha) CI: AUC ± z_{1−α/2}·√Var_DeLong, обрізаний у [0, 1]."""
    aucs, cov = fast_delong(y, score)
    auc = float(aucs[0])
    half = float(norm.ppf(1 - alpha / 2)) * math.sqrt(cov[0, 0])
    return auc, max(0.0, auc - half), min(1.0, auc + half)


def _se_and_z(diff: float, var_diff: float) -> tuple[float, float]:
    """SE різниці AUC і z = diff / SE; за нульової дисперсії z = 0 (diff = 0) або ±inf."""
    se = math.sqrt(var_diff) if var_diff > 0 else 0.0
    # var_diff = 0: однакові ранги → z = 0, p = 1; різні AUC без дисперсії → z = ±inf, p = 0
    z = diff / se if se > 0 else (0.0 if diff == 0 else math.copysign(math.inf, diff))
    return se, z


def delong_test(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, alpha: float = ALPHA
) -> DelongResult:
    """H0: AUC_a = AUC_b на тих самих спостереженнях. z = (AUC_a − AUC_b) / SE, p двобічний."""
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    aucs, cov = fast_delong(y, np.vstack([a, b]))
    diff = float(aucs[0] - aucs[1])
    # Var(AUC_a − AUC_b) = Var_a + Var_b − 2·Cov_ab
    se, z = _se_and_z(diff, float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    p = float(2 * norm.sf(abs(z)))
    half = float(norm.ppf(1 - alpha / 2)) * se
    return DelongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        diff=diff,
        se_diff=se,
        z=z,
        p=p,
        ci_low=diff - half,
        ci_high=diff + half,
    )
