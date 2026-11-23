"""Поріг класифікації за індексом Юдена (на OOF-прогнозах навчальної частини)."""

import numpy as np
from sklearn.metrics import roc_curve

# TPR, FPR — дроби k/P, l/N; рівні в точній арифметиці J у float можуть різнитися на ~1e-16
# (напр., 2/3 − 0 ≠ 1 − 1/3), тому максимуми порівнюємо з допуском.
J_ATOL = 1e-12


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Поріг, що максимізує J = TPR − FPR (= Sens + Spec − 1); з кількох максимумів — найменший.

    Прогноз позитивний, якщо p ≥ поріг (семантика roc_curve). Поріг inf від sklearn ігнорується.
    """
    y_true = np.asarray(y).ravel() == 1
    fpr, tpr, thr = roc_curve(y_true, np.asarray(p, dtype=float).ravel(), drop_intermediate=False)
    finite = np.isfinite(thr)
    j, thr = tpr[finite] - fpr[finite], thr[finite]
    return float(thr[j >= j.max() - J_ATOL].min())
