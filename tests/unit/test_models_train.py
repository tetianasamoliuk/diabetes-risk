"""Тести навчання, оцінки на тесті, збереження та запобіжників витоку."""

import json
import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import risk.models.train as train
from risk.config import FEATURES
from risk.errors import ForbiddenFeatureError, ModelNotTrainedError
from risk.models import pipelines
from risk.models.pipelines import cv_splitter, make_logit

N = 500


def _synthetic_cohort(n: int = N, seed: int = 0) -> pd.DataFrame:
    """Когорта з колонками harmonize: ризик зростає з віком, ІМТ, талією; ~25% позитивних."""
    rng = np.random.default_rng(seed)
    age, bmi = rng.uniform(20, 80, n), rng.normal(29, 6, n)
    df = pd.DataFrame(
        {
            "SEQN": np.arange(10_000, 10_000 + n),
            "age": age,
            "male": rng.integers(0, 2, n).astype(float),
            "bmi": bmi,
            "waist": 60 + 1.2 * bmi + rng.normal(0, 8, n),
            "sbp": rng.normal(125, 15, n),
            "dbp": rng.normal(72, 10, n),
            "tc": rng.normal(190, 35, n),
            "hdl": rng.normal(52, 14, n),
            "pa_min_week": rng.choice([0.0, 120.0, 300.0, 600.0], n),
            "fam_dm": rng.integers(0, 2, n).astype(float),
            "bp_med": rng.integers(0, 2, n).astype(float),
            "cycle_2017": rng.integers(0, 2, n).astype(float),
        }
    )
    df["pa_h_week"] = df["pa_min_week"] / 60
    logit = -1.6 + 0.06 * (age - 50) + 0.1 * (bmi - 29) + 0.8 * df["fam_dm"]
    df["t2d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return _add_diagnosis(df, rng)


def _add_diagnosis(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """diq010 (80% хворих діагностовані), t2d_undiag і high_glu_hist лише для недіагностованих."""
    n = len(df)
    diag = (df["t2d"] == 1) & (rng.random(n) < 0.8)
    df["diq010"] = np.where(diag, 1.0, 2.0)
    df["t2d_undiag"] = df["t2d"].where(~diag)
    glu = (rng.random(n) < 0.15 + 0.4 * df["t2d"]).astype(float)
    df["high_glu_hist"] = glu.where(~diag)
    df.loc[df.index % 11 == 0, "sbp"] = np.nan  # пропуски поза пунктами FINDRISC
    df.loc[df.index % 13 == 0, "hdl"] = np.nan
    return df


# Ті самі пайплайни, менше дерев / ітерацій — для швидкості юніт-тестів
FAST_MAKERS = {
    "rf": lambda: pipelines.make_rf().set_params(model__n_estimators=30),
    "hgb": lambda: pipelines.make_hgb().set_params(model__max_iter=30),
}


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return _synthetic_cohort()


@pytest.fixture(scope="module")
def trained(cohort: pd.DataFrame) -> train.TrainResult:
    with pytest.MonkeyPatch.context() as mp:
        for name, make in FAST_MAKERS.items():
            mp.setitem(pipelines.MAKERS, name, make)
        return train.train_all(cohort)


@pytest.fixture(scope="module")
def metrics(trained: train.TrainResult, cohort: pd.DataFrame) -> dict:
    return train.evaluate_on_test(trained, cohort)


@pytest.fixture(scope="module")
def predictions(trained: train.TrainResult, cohort: pd.DataFrame) -> pd.DataFrame:
    return train.test_predictions_frame(trained, cohort)


# --- запобіжник ознак у split ---


@pytest.mark.parametrize("bad", [["age", "LBXGH"], ["hba1c"], ["DIQ010"], ["t2d"]])
def test_split_rejects_forbidden_features(cohort: pd.DataFrame, bad: list[str]) -> None:
    df = cohort.assign(LBXGH=5.5, hba1c=5.5, DIQ010=2.0)
    with pytest.raises(ForbiddenFeatureError):
        train.split(df, features=bad)


# --- спліт без перетину, стратифікований ---


def test_split_disjoint_and_stratified(cohort: pd.DataFrame) -> None:
    X_tr, X_te, y_tr, y_te = train.split(cohort)
    assert X_tr.index.intersection(X_te.index).empty
    assert X_tr.index.union(X_te.index).equals(cohort.index)
    assert len(X_te) == math.ceil(0.2 * len(cohort))
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert list(X_tr.columns) == FEATURES
    assert (y_tr.index == X_tr.index).all()


def test_split_is_deterministic(cohort: pd.DataFrame) -> None:
    assert train.split(cohort)[1].index.equals(train.split(cohort)[1].index)


# --- fit_model / oof ---


def test_fit_model_info(cohort: pd.DataFrame) -> None:
    X, _, y, _ = train.split(cohort)
    est, info = train.fit_model("logit", X, y, cv_splitter())
    assert isinstance(est, Pipeline)
    assert info["best_params"]["model__C"] in (0.1, 1, 10)
    assert len(info["cv_folds"]) == 5
    assert info["cv_auc_mean"] == pytest.approx(np.mean(info["cv_folds"]))
    assert info["cv_auc_sd"] == pytest.approx(np.std(info["cv_folds"], ddof=1))
    assert est.predict_proba(X).shape == (len(X), 2)  # refit на всьому train


def test_oof_predictions_do_not_touch_estimator(cohort: pd.DataFrame) -> None:
    X, _, y, _ = train.split(cohort)
    est = make_logit()
    oof = train.oof_predictions(est, X, y, cv_splitter())
    assert oof.shape == (len(X),)
    assert ((oof > 0) & (oof < 1)).all()
    assert not hasattr(est[-1], "coef_")  # навчено клон, а не переданий об'єкт


# --- fit_model бачить лише train ---


def test_train_all_fits_only_on_train(
    cohort: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[pd.Index] = []

    def spy(name: str, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> tuple[Pipeline, dict]:
        seen.append(X.index)
        return make_logit().fit(X, y), {"best_params": {}}

    monkeypatch.setattr(train, "fit_model", spy)
    result = train.train_all(cohort)
    n_train = len(cohort) - math.ceil(0.2 * len(cohort))
    assert [len(i) for i in seen] == [n_train] * 3
    assert all(i.intersection(result.test_index).empty for i in seen)


# --- TrainResult ---


def test_train_result_structure(trained: train.TrainResult, cohort: pd.DataFrame) -> None:
    names = ["models", "thresholds", "cv", "oof", "test_index", "train_index"]
    assert [f.name for f in fields(trained)] == names
    assert set(trained.models) == {"logit", "rf", "hgb"}
    assert trained.train_index.intersection(trained.test_index).empty
    assert len(trained.train_index) + len(trained.test_index) == len(cohort)
    for name in trained.models:
        assert trained.oof[name].shape == (len(trained.train_index),)
        assert 0 < trained.thresholds[name] < 1


def test_thresholds_are_youden_on_oof(trained: train.TrainResult, cohort: pd.DataFrame) -> None:
    from risk.models.threshold import youden_threshold

    y_tr = cohort.loc[trained.train_index, "t2d"]
    for name, oof in trained.oof.items():
        assert trained.thresholds[name] == youden_threshold(y_tr, oof)


# --- evaluate_on_test: FINDRISC (c) не найкращий ---


def test_evaluate_best_model_and_findrisc_c_invalid(metrics: dict) -> None:
    assert metrics["best_model"] in {"logit", "rf", "hgb"}
    cv_auc = {n: c["cv_auc_mean"] for n, c in metrics["cv"].items()}
    assert metrics["best_model"] == max(cv_auc, key=cv_auc.get)  # за CV, не за тестом
    assert metrics["best_model_rule"] == "cv_auc_mean"
    test_auc = {n: m["auc"] for n, m in metrics["test"].items()}
    assert metrics["test_auc_rank"] == sorted(test_auc, key=test_auc.get, reverse=True)
    assert metrics["findrisc"]["c"]["valid"] is False
    assert metrics["findrisc"]["a"]["valid"] is True
    assert metrics["findrisc"]["b"]["valid"] is True


def test_best_model_ignores_test_auc(
    trained: train.TrainResult, metrics: dict, cohort: pd.DataFrame
) -> None:
    """Найгірша на тесті модель з найвищим CV AUC стає best; FINDRISC рахується проти неї."""
    worst_on_test = metrics["test_auc_rank"][-1]
    cv = {n: {**c, "cv_auc_mean": float(n == worst_on_test)} for n, c in trained.cv.items()}
    m = train.evaluate_on_test(replace(trained, cv=cv), cohort)
    assert m["best_model"] == worst_on_test
    assert {r["model"] for r in m["delong_vs_findrisc"]} == {worst_on_test}


def test_evaluate_keys_and_sizes(metrics: dict, cohort: pd.DataFrame) -> None:
    expected = {"test", "cv", "delong", "best_model", "findrisc", "delong_vs_findrisc"}
    expected |= {"calibration", "n_train", "n_test", "n_pos_train", "n_pos_test", "features"}
    expected |= {"thresholds", "best_params", "versions", "trained_at"}
    expected |= {"best_model_rule", "test_auc_rank"}
    assert expected <= set(metrics)
    assert metrics["n_train"] + metrics["n_test"] == len(cohort)
    assert metrics["n_pos_train"] + metrics["n_pos_test"] == cohort["t2d"].sum()
    assert metrics["features"] == FEATURES
    assert set(metrics["versions"]) == {"python", "sklearn", "pandas", "numpy", "shap"}
    assert {m["n"] for m in metrics["test"].values()} == {metrics["n_test"]}


def test_evaluate_delong_pairs(metrics: dict) -> None:
    pairs = [(r["a"], r["b"]) for r in metrics["delong"]]
    assert pairs == [("logit", "rf"), ("logit", "hgb"), ("rf", "hgb")]
    for r in metrics["delong"]:
        assert r["auc_a"] == pytest.approx(metrics["test"][r["a"]]["auc"])
        assert 0 <= r["p"] <= 1


def test_findrisc_only_on_test_rows(
    metrics: dict, trained: train.TrainResult, cohort: pd.DataFrame
) -> None:
    test = cohort.loc[trained.test_index]
    undiag = test["diq010"] != 1
    assert metrics["findrisc"]["a"]["n"] == len(test)  # у синтетиці пункти без пропусків
    assert metrics["findrisc"]["b"]["n"] == int(undiag.sum())
    assert metrics["findrisc"]["b"]["n_pos"] == int(test.loc[undiag, "t2d_undiag"].sum())
    assert metrics["findrisc"]["c"]["auc"] > metrics["findrisc"]["a"]["auc"]  # витік завищує


def test_delong_vs_findrisc_rows(metrics: dict) -> None:
    rows = {r["variant"]: r for r in metrics["delong_vs_findrisc"]}
    assert set(rows) == {"a", "b"}
    best = metrics["best_model"]
    assert all(r["model"] == best for r in rows.values())
    assert rows["a"]["auc_a"] == pytest.approx(metrics["test"][best]["auc"])
    for key in ("a", "b"):
        assert rows[key]["auc_b"] == pytest.approx(metrics["findrisc"][key]["auc"])
        assert rows[key]["n"] == metrics["findrisc"][key]["n"]


def test_calibration_tables(metrics: dict) -> None:
    for table in metrics["calibration"].values():
        assert set(table) == {"bin", "mean_pred", "frac_pos", "n"}
        assert sum(table["n"]) == metrics["n_test"]


# --- test_predictions_frame ---


def test_predictions_frame(
    predictions: pd.DataFrame, trained: train.TrainResult, cohort: pd.DataFrame
) -> None:
    cols = ["SEQN", "t2d", "t2d_undiag", "diq010", "p_logit", "p_rf", "p_hgb"]
    assert list(predictions.columns) == [*cols, "findrisc_a", "findrisc_b"]
    assert predictions.index.equals(trained.test_index)
    assert predictions["SEQN"].equals(cohort.loc[trained.test_index, "SEQN"])
    diag = predictions["diq010"] == 1
    assert predictions.loc[diag, "findrisc_b"].isna().all()
    assert predictions.loc[~diag, "findrisc_b"].notna().all()


def test_predictions_match_metrics(predictions: pd.DataFrame, metrics: dict) -> None:
    from sklearn.metrics import roc_auc_score

    for name in ("logit", "rf", "hgb"):
        auc = roc_auc_score(predictions["t2d"], predictions[f"p_{name}"])
        assert auc == pytest.approx(metrics["test"][name]["auc"])


# --- save / load ---


def test_save_and_load_roundtrip(
    trained: train.TrainResult,
    metrics: dict,
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train.save(trained, metrics, predictions, models_dir=tmp_path)
    names = {"logit.joblib", "rf.joblib", "hgb.joblib", "metrics.json", "test_predictions.parquet"}
    assert names <= {p.name for p in tmp_path.iterdir()}
    models, loaded = train.load_models(tmp_path)
    assert loaded == json.loads(json.dumps(metrics, default=float))
    X_te = cohort.loc[trained.test_index, FEATURES]
    for name, model in models.items():
        expected = trained.models[name].predict_proba(X_te)
        np.testing.assert_allclose(model.predict_proba(X_te), expected)
    pd.testing.assert_frame_equal(train.load_test_predictions(tmp_path), predictions)


def test_metrics_json_is_utf8_readable(
    trained: train.TrainResult, metrics: dict, predictions: pd.DataFrame, tmp_path: Path
) -> None:
    train.save(trained, metrics, predictions, models_dir=tmp_path / "new")
    text = (tmp_path / "new" / "metrics.json").read_text(encoding="utf-8")
    assert json.loads(text)["best_model"] == metrics["best_model"]


def test_load_models_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotTrainedError, match="train_models.py"):
        train.load_models(tmp_path)


def test_load_test_predictions_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotTrainedError):
        train.load_test_predictions(tmp_path)
