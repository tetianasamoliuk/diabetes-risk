"""Узгодженість закомічених артефактів: metrics.json, прогнози, моделі, когорта, числа."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from risk.config import FEATURES, MODELS_DIR, PROCESSED_DIR

MODELS = ["logit", "rf", "hgb"]


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", MODELS)
def test_test_predictions_auc_matches_metrics(name: str, metrics: dict) -> None:
    preds = pd.read_parquet(MODELS_DIR / "test_predictions.parquet")
    auc = roc_auc_score(preds["t2d"], preds[f"p_{name}"])
    assert auc == pytest.approx(metrics["test"][name]["auc"], abs=1e-9)


@pytest.mark.parametrize("name", MODELS)
def test_model_expects_config_features(name: str) -> None:
    model = joblib.load(MODELS_DIR / f"{name}.joblib")
    assert list(model.feature_names_in_) == FEATURES


def test_split_sizes_match_cohort(metrics: dict) -> None:
    cohort = pd.read_parquet(PROCESSED_DIR / "cohort.parquet")
    assert metrics["n_train"] + metrics["n_test"] == len(cohort)


def test_best_model_consistent(metrics: dict) -> None:
    numbers = json.loads((PROCESSED_DIR / "results_numbers.json").read_text(encoding="utf-8"))
    cv = {m: metrics["cv"][m]["cv_auc_mean"] for m in MODELS}
    assert metrics["best_model"] == max(cv, key=cv.get)
    assert metrics["best_model"] == numbers["best_model"]


def test_metrics_features_are_config_features(metrics: dict) -> None:
    assert metrics["features"] == FEATURES


def _read_numbers() -> dict:
    return json.loads((PROCESSED_DIR / "results_numbers.json").read_text(encoding="utf-8"))


def check_numbers(numbers: dict, metrics: dict, n_cohort: int, n_pos: int) -> int:
    """Звіряє числа results_numbers.json з metrics.json і когортою; повертає кількість перевірок."""
    expected = {
        "n_cohort": n_cohort,
        "n_pos": n_pos,
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }
    for name in MODELS:
        expected[f"cv_auc_{name}"] = round(metrics["cv"][name]["cv_auc_mean"], 3)
        expected[f"test_auc_{name}"] = round(metrics["test"][name]["auc"], 3)
    for variant in "abc":
        expected[f"findrisc_auc_{variant}"] = round(metrics["findrisc"][variant]["auc"], 3)
    for key, value in expected.items():
        assert numbers[key] == pytest.approx(value, abs=1e-9), key
    assert numbers["n_train"] + numbers["n_test"] == numbers["n_cohort"]
    assert metrics["n_pos_train"] + metrics["n_pos_test"] == numbers["n_pos"]
    return len(expected) + 2


def test_results_numbers_match_artifacts(metrics: dict) -> None:
    cohort = pd.read_parquet(PROCESSED_DIR / "cohort.parquet")
    check_numbers(_read_numbers(), metrics, len(cohort), int(cohort["t2d"].sum()))


@pytest.mark.parametrize("key", ["test_auc_hgb", "cv_auc_logit", "findrisc_auc_b", "n_test"])
def test_numbers_guard_fails_on_tampering(key: str, metrics: dict, tmp_path: Path) -> None:
    """Копія results_numbers.json зі зміненим значенням має валити звірку."""
    cohort = pd.read_parquet(PROCESSED_DIR / "cohort.parquet")
    copy = tmp_path / "results_numbers.json"
    copy.write_text(json.dumps(_read_numbers() | {key: 0.5}), encoding="utf-8")
    tampered = json.loads(copy.read_text(encoding="utf-8"))
    with pytest.raises(AssertionError):
        check_numbers(tampered, metrics, len(cohort), int(cohort["t2d"].sum()))
