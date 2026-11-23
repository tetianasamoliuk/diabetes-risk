"""SHAP-пояснення HistGradientBoosting: TreeExplainer на сирих значеннях з NaN.

Значення — внески у log-odds класу 1: base_values + Σ values = decision_function моделі.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from risk.errors import DataValidationError


def make_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """TreeExplainer для останнього кроку; пайплайн з препроцесингом (імпутер тощо) відхиляється."""
    if len(pipeline.steps) > 1:
        names = ", ".join(name for name, _ in pipeline.steps[:-1])
        raise DataValidationError(
            f"Пояснення мають бути на сирих значеннях, а пайплайн має препроцесинг ({names}); "
            "SHAP рахується для HistGradientBoosting без імпутації"
        )
    return shap.TreeExplainer(pipeline[-1])


def global_shap(explainer: shap.TreeExplainer, X: pd.DataFrame) -> shap.Explanation:
    """SHAP для всіх рядків X: values (n, k), base_values (n,), назви ознак — колонки X."""
    expl = explainer(X)
    expl.feature_names = list(X.columns)
    return expl


def local_shap(explainer: shap.TreeExplainer, row: pd.DataFrame) -> shap.Explanation:
    """SHAP для однієї людини: values (k,), base_values — скаляр (для waterfall)."""
    if len(row) != 1:
        raise DataValidationError(f"Очікується один рядок, отримано {len(row)}")
    return global_shap(explainer, row)[0]


def top_contributions(expl: shap.Explanation, k: int = 3) -> list[tuple[str, float]]:
    """Перші k ознак за спаданням |внеску| з однорядкового Explanation."""
    pairs = zip(expl.feature_names, np.asarray(expl.values, dtype=float), strict=True)
    return sorted(((f, float(v)) for f, v in pairs), key=lambda t: abs(t[1]), reverse=True)[:k]
