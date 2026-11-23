"""Навчання на реальній когорті: очікувані діапазони AUC, запобіжники, артефакти models/.

Числа поза діапазоном — привід розібратися, а не підганяти тест.
"""

from pathlib import Path

import pandas as pd
import pytest

import risk.models.train as train
from risk.config import MODELS_DIR, PROCESSED_DIR

pytestmark = pytest.mark.slow

MODELS = ("logit", "rf", "hgb")
ARTIFACTS = [f"{m}.joblib" for m in MODELS] + ["metrics.json", "test_predictions.parquet"]


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "cohort.parquet")


@pytest.fixture(scope="module")
def result(cohort: pd.DataFrame) -> train.TrainResult:
    return train.train_all(cohort)


@pytest.fixture(scope="module")
def metrics(result: train.TrainResult, cohort: pd.DataFrame) -> dict:
    return train.evaluate_on_test(result, cohort)


@pytest.fixture(scope="module")
def saved_dir(
    result: train.TrainResult,
    metrics: dict,
    cohort: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    out = tmp_path_factory.mktemp("models")
    train.save(result, metrics, train.test_predictions_frame(result, cohort), out)
    return out


def test_test_auc_in_expected_range(metrics: dict) -> None:
    for name in MODELS:
        assert 0.78 <= metrics["test"][name]["auc"] <= 0.88, name


def test_findrisc_auc(metrics: dict) -> None:
    fr = metrics["findrisc"]
    assert 0.74 <= fr["a"]["auc"] <= 0.84
    assert fr["c"]["auc"] > fr["a"]["auc"]  # витік завищує AUC
    assert fr["c"]["valid"] is False


def test_best_model_is_ml_chosen_by_cv(metrics: dict) -> None:
    assert metrics["best_model"] in MODELS
    cv_auc = {n: metrics["cv"][n]["cv_auc_mean"] for n in MODELS}
    assert metrics["best_model"] == max(cv_auc, key=cv_auc.get)  # не за тестом
    assert metrics["best_model_rule"] == "cv_auc_mean"
    assert {r["model"] for r in metrics["delong_vs_findrisc"]} == {metrics["best_model"]}


def test_thresholds_reasonable(result: train.TrainResult) -> None:
    for name in MODELS:
        assert 0.05 < result.thresholds[name] < 0.6, name


def test_train_test_disjoint(result: train.TrainResult, cohort: pd.DataFrame) -> None:
    assert result.train_index.intersection(result.test_index).empty
    assert len(result.train_index) + len(result.test_index) == len(cohort)


def test_artifacts_saved_and_loadable(saved_dir: Path) -> None:
    assert all((saved_dir / f).exists() for f in ARTIFACTS)
    models, loaded = train.load_models(saved_dir)
    assert set(models) == set(MODELS)
    assert loaded["best_model"] in MODELS


def test_committed_artifacts_match_fresh_run(metrics: dict, result: train.TrainResult) -> None:
    """models/ з scripts/train_models.py відтворюється: ті самі test AUC, пороги і тестові рядки."""
    assert all((MODELS_DIR / f).exists() for f in ARTIFACTS)
    _, committed = train.load_models(MODELS_DIR)
    for name in MODELS:
        assert committed["test"][name]["auc"] == pytest.approx(metrics["test"][name]["auc"])
        assert committed["thresholds"][name] == pytest.approx(result.thresholds[name])
    assert train.load_test_predictions(MODELS_DIR).index.equals(result.test_index)


def test_statsmodels_logit_matches_sklearn_auc(
    result: train.TrainResult, cohort: pd.DataFrame
) -> None:
    """Формула statsmodels і прогнозний sklearn-логіт мають близький AUC (< 0.01)."""
    import numpy as np
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    from risk.config import FEATURES

    train_df, test_df = cohort.loc[result.train_index], cohort.loc[result.test_index]
    d = train_df[[*FEATURES, "t2d"]].dropna()
    fit = sm.Logit(d["t2d"], sm.add_constant(d[FEATURES].astype(float))).fit(disp=0)
    t = test_df[[*FEATURES, "t2d"]].dropna()
    p_sm = fit.predict(sm.add_constant(t[FEATURES].astype(float), has_constant="add"))
    p_sk = result.models["logit"].predict_proba(t[FEATURES])[:, 1]
    auc_sm, auc_sk = roc_auc_score(t["t2d"], p_sm), roc_auc_score(t["t2d"], p_sk)
    assert np.isfinite(auc_sm) and abs(auc_sm - auc_sk) < 0.01, (
        f"AUC statsmodels={auc_sm:.4f} sklearn={auc_sk:.4f} n={len(t)}"
    )
