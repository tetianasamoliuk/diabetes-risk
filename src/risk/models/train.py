"""Навчання трьох моделей, оцінка на hold-out тесті, збереження артефактів.

Схема без витоку: stratified hold-out 20% відкладається ДО будь-якого fit; GridSearchCV,
OOF-прогнози і поріг Юдена — лише на train; тест (і FINDRISC на ньому) не бере участі в підборі
параметрів, порогів і виборі основної моделі, його оцінює лише evaluate_on_test.
"""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from risk.config import FEATURES, MODELS_DIR, SEED, TEST_SIZE
from risk.errors import ModelNotTrainedError
from risk.findrisc.score import FindriscVariant, variants
from risk.models.calibration import calibration_table
from risk.models.delong import auc_ci, delong_test
from risk.models.evaluate import classification_metrics
from risk.models.pipelines import MAKERS, PARAM_GRIDS, assert_no_forbidden_features, cv_splitter
from risk.models.threshold import youden_threshold

PAIRS = [("logit", "rf"), ("logit", "hgb"), ("rf", "hgb")]
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "test_predictions.parquet"


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Навчені пайплайни, пороги, результати CV і OOF-прогнози; тест ще не оцінено."""

    models: dict[str, Pipeline]
    thresholds: dict[str, float]  # Юден на OOF-прогнозах train
    cv: dict[str, dict]
    oof: dict[str, np.ndarray]
    test_index: pd.Index  # індекс аналітичної вибірки
    train_index: pd.Index


# ---------- навчання ----------
def split(
    cohort: pd.DataFrame,
    features: list[str] = FEATURES,
    outcome: str = "t2d",
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out: X_train, X_test, y_train, y_test з індексом когорти."""
    assert_no_forbidden_features(features)
    X, y = cohort[list(features)], cohort[outcome].astype(int)
    return tuple(train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed))


def fit_model(
    name: str, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator
) -> tuple[Pipeline, dict]:
    """GridSearchCV за ROC-AUC; refit найкращих параметрів на всьому X."""
    search = GridSearchCV(
        MAKERS[name](), PARAM_GRIDS[name], scoring="roc_auc", cv=cv, refit=True, n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_, _cv_info(search)


def _cv_info(search: GridSearchCV) -> dict:
    """AUC по фолдах для найкращих параметрів; SD вибірковий (ddof=1)."""
    i, res = search.best_index_, search.cv_results_
    folds = [float(res[f"split{k}_test_score"][i]) for k in range(search.n_splits_)]
    return {
        "best_params": search.best_params_,
        "cv_auc_mean": float(np.mean(folds)),
        "cv_auc_sd": float(np.std(folds, ddof=1)),
        "cv_folds": folds,
    }


def oof_predictions(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator
) -> np.ndarray:
    """Out-of-fold ймовірності класу 1: кожен рядок прогнозує модель, що його не бачила."""
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def train_all(cohort: pd.DataFrame, features: list[str] = FEATURES) -> TrainResult:
    """Спліт → для кожної моделі GridSearchCV, OOF і поріг Юдена — лише на train."""
    X_tr, X_te, y_tr, _ = split(cohort, features)
    cv = cv_splitter()
    models, cv_info, oof, thresholds = {}, {}, {}, {}
    for name in MAKERS:
        models[name], cv_info[name] = fit_model(name, X_tr, y_tr, cv)
        oof[name] = oof_predictions(models[name], X_tr, y_tr, cv)
        thresholds[name] = youden_threshold(y_tr, oof[name])
    return TrainResult(models, thresholds, cv_info, oof, X_te.index, X_tr.index)


# ---------- оцінка на тесті ----------
def _test_probas(
    result: TrainResult, test: pd.DataFrame, features: list[str]
) -> dict[str, pd.Series]:
    X = test[list(features)]
    return {n: pd.Series(m.predict_proba(X)[:, 1], index=X.index) for n, m in result.models.items()}


def _delong_row(y: pd.Series, probas: dict[str, pd.Series], a: str, b: str) -> dict:
    return {"a": a, "b": b, **asdict(delong_test(y, probas[a], probas[b]))}


def _findrisc_auc(v: FindriscVariant) -> dict:
    auc, low, high = auc_ci(v.y, v.score)
    return {
        "auc": auc,
        "ci_low": low,
        "ci_high": high,
        "n": len(v.y),
        "n_pos": int(v.y.sum()),
        "valid": v.is_valid_for_comparison,
    }


def _vs_findrisc(p_best: pd.Series, best: str, v: FindriscVariant) -> dict:
    """DeLong модель vs FINDRISC на рядках варіанта і з його ціллю (a: t2d, b: t2d_undiag)."""
    res = delong_test(v.y, p_best.loc[v.score.index], v.score)
    return {"model": best, "variant": v.key, "n": len(v.y), **asdict(res)}


def _calibration(y: pd.Series, p: pd.Series) -> dict[str, list]:
    return calibration_table(y, p).to_dict(orient="list")


def _versions() -> dict[str, str]:
    libs = {"sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "shap": "shap"}
    return {"python": platform.python_version(), **{k: version(v) for k, v in libs.items()}}


def _meta(result: TrainResult, cohort: pd.DataFrame, features: list[str]) -> dict:
    y = cohort["t2d"]
    return {
        "n_train": len(result.train_index),
        "n_test": len(result.test_index),
        "n_pos_train": int(y.loc[result.train_index].sum()),
        "n_pos_test": int(y.loc[result.test_index].sum()),
        "features": list(features),
        "thresholds": result.thresholds,
        "best_params": {n: c["best_params"] for n, c in result.cv.items()},
        "versions": _versions(),
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def evaluate_on_test(
    result: TrainResult, cohort: pd.DataFrame, features: list[str] = FEATURES
) -> dict:
    """Метрики на hold-out тесті, DeLong, FINDRISC (лише тестові рядки), калібрування, метадані.

    Основна модель — за CV AUC на train, бо вибір за test AUC був би відбором на тесті (витік).
    """
    test = cohort.loc[result.test_index]
    y = test["t2d"].astype(int)
    probas = _test_probas(result, test, features)
    metrics = {n: classification_metrics(y, p, result.thresholds[n]) for n, p in probas.items()}
    best = max(result.cv, key=lambda n: result.cv[n]["cv_auc_mean"])
    fr = variants(test)
    return {
        "test": {n: asdict(m) for n, m in metrics.items()},
        "cv": result.cv,
        "delong": [_delong_row(y, probas, a, b) for a, b in PAIRS],
        "best_model": best,
        "best_model_rule": "cv_auc_mean",
        "test_auc_rank": sorted(metrics, key=lambda n: metrics[n].auc, reverse=True),  # довідково
        "findrisc": {k: _findrisc_auc(v) for k, v in fr.items()},
        "delong_vs_findrisc": [
            _vs_findrisc(probas[best], best, v) for v in fr.values() if v.is_valid_for_comparison
        ],
        "calibration": {n: _calibration(y, p) for n, p in probas.items()},
        **_meta(result, cohort, features),
    }


def test_predictions_frame(
    result: TrainResult, cohort: pd.DataFrame, features: list[str] = FEATURES
) -> pd.DataFrame:
    """Тестові прогнози моделей і бали FINDRISC (a)/(b) — для ROC/калібрування в застосунку."""
    test = cohort.loc[result.test_index]
    out = test[["SEQN", "t2d", "t2d_undiag", "diq010"]].copy()
    for name, p in _test_probas(result, test, features).items():
        out[f"p_{name}"] = p
    fr = variants(test)
    out["findrisc_a"], out["findrisc_b"] = fr["a"].score, fr["b"].score  # NaN поза варіантом
    return out


# ---------- артефакти ----------
def save(
    result: TrainResult, metrics: dict, predictions: pd.DataFrame, models_dir: Path = MODELS_DIR
) -> None:
    """models/{logit,rf,hgb}.joblib, metrics.json, test_predictions.parquet."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in result.models.items():
        joblib.dump(model, models_dir / f"{name}.joblib", compress=3)
    text = json.dumps(metrics, indent=2, ensure_ascii=False, default=float)
    (models_dir / METRICS_FILE).write_text(text, encoding="utf-8")
    predictions.to_parquet(models_dir / PREDICTIONS_FILE)


def _require(paths: list[Path]) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ModelNotTrainedError(
            f"Немає збережених моделей ({', '.join(missing)}) — виконайте scripts/train_models.py"
        )


def load_models(models_dir: Path = MODELS_DIR) -> tuple[dict[str, Pipeline], dict]:
    """Збережені пайплайни і metrics.json; ModelNotTrainedError, якщо файлів немає."""
    paths = {name: models_dir / f"{name}.joblib" for name in MAKERS}
    _require([*paths.values(), models_dir / METRICS_FILE])
    models = {name: joblib.load(p) for name, p in paths.items()}
    return models, json.loads((models_dir / METRICS_FILE).read_text(encoding="utf-8"))


def load_test_predictions(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Тестові прогнози (індекс — індекс когорти); ModelNotTrainedError, якщо файлу немає."""
    _require([models_dir / PREDICTIONS_FILE])
    return pd.read_parquet(models_dir / PREDICTIONS_FILE)
