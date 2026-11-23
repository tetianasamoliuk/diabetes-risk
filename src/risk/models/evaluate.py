"""Метрики класифікації за фіксованим порогом, їхні довірчі інтервали і калібрування на тесті."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint

from risk.config import ALPHA
from risk.models.calibration import brier, calibration_slope_intercept
from risk.models.delong import auc_ci


@dataclass(frozen=True, slots=True)
class Metrics:
    """Метрики на тесті; *_ci_* — 95 % CI Вілсона, calib_* — нахил і зсув калібрування."""

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float
    n: int
    se_ci_low: float
    se_ci_high: float
    sp_ci_low: float
    sp_ci_high: float
    ppv_ci_low: float
    ppv_ci_high: float
    calib_slope: float
    calib_intercept: float


def _ratio(num: int, den: int) -> float:
    """Частка num / den; 0.0 при нульовому знаменнику (як zero_division=0 у sklearn)."""
    return num / den if den else 0.0


def wilson_ci(count: int, n: int) -> tuple[float, float]:
    """95 % CI Вілсона для частки count / n; без спостережень — (NaN, NaN)."""
    if not n:
        return float("nan"), float("nan")
    low, high = proportion_confint(count, n, alpha=ALPHA, method="wilson")
    return float(low), float(high)


def ppv_at_prevalence(se: float, sp: float, prev: float) -> float:
    """PPV за теоремою Баєса для поширеності prev (усі три — частки 0–1).

    PPV = Se·π / (Se·π + (1 − Sp)·(1 − π)): за тих самих Se і Sp PPV падає разом з π.
    """
    tp = se * prev
    return tp / (tp + (1 - sp) * (1 - prev))


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Metrics:
    """Метрики при правилі «позитивний, якщо p ≥ threshold»; AUC з 95% CI (DeLong)."""
    pos = np.asarray(y).ravel() == 1
    prob = np.asarray(p, dtype=float).ravel()
    pred = prob >= threshold
    tp, fp = int(np.sum(pred & pos)), int(np.sum(pred & ~pos))
    fn, tn = int(np.sum(~pred & pos)), int(np.sum(~pred & ~pos))
    auc, low, high = auc_ci(pos, prob)
    se_ci, sp_ci, ppv_ci = wilson_ci(tp, tp + fn), wilson_ci(tn, tn + fp), wilson_ci(tp, tp + fp)
    slope, intercept = calibration_slope_intercept(pos, prob)
    return Metrics(
        accuracy=(tp + tn) / pos.size, precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp), f1=_ratio(2 * tp, 2 * tp + fp + fn),
        auc=auc, auc_ci_low=low, auc_ci_high=high, brier=brier(pos, prob),
        tn=tn, fp=fp, fn=fn, tp=tp, threshold=float(threshold), n=int(pos.size),
        se_ci_low=se_ci[0], se_ci_high=se_ci[1], sp_ci_low=sp_ci[0], sp_ci_high=sp_ci[1],
        ppv_ci_low=ppv_ci[0], ppv_ci_high=ppv_ci[1], calib_slope=slope, calib_intercept=intercept,
    )  # fmt: skip


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Точки ROC-кривої (fpr, tpr) для графіка."""
    fpr, tpr, _ = roc_curve(np.asarray(y).ravel() == 1, np.asarray(p, dtype=float).ravel())
    return fpr, tpr
