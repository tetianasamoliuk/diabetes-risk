"""Запобіжники методології: швидкі перевірки без реального навчання."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from risk.config import DISCLAIMER, FEATURES, FORBIDDEN_FEATURE_PATTERNS, MODELS_DIR
from risk.errors import ForbiddenFeatureError
from risk.findrisc.score import variants
from risk.models.pipelines import MAKERS, assert_no_forbidden_features
from risk.models.summary import findrisc_vs_model_table
from risk.models.train import METRICS_FILE, TrainResult, evaluate_on_test, split

MODELS = ["logit", "rf", "hgb"]


@pytest.fixture(scope="module")
def metrics() -> dict:
    path = MODELS_DIR / METRICS_FILE
    if not path.exists():
        pytest.fail("артефакти відсутні: make train")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def synth() -> pd.DataFrame:
    """Синтетична когорта (1000 рядків, ~15% позитивних): колонки FEATURES і FINDRISC."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "male": rng.integers(0, 2, n),
            "bmi": rng.normal(28, 5, n).clip(15, 60),
            "waist": rng.normal(95, 12, n).clip(60, 160),
            "sbp": rng.normal(122, 15, n),
            "dbp": rng.normal(72, 10, n),
            "tc": rng.normal(190, 35, n),
            "hdl": rng.normal(52, 14, n),
            "pa_h_week": rng.exponential(3, n),
            "fam_dm": rng.integers(0, 2, n),
            "bp_med": rng.integers(0, 2, n),
            "cycle_2017": rng.integers(0, 2, n),
            "high_glu_hist": rng.integers(0, 2, n),
        }
    )
    df["pa_min_week"] = df["pa_h_week"] * 60
    df["t2d"] = (rng.random(n) < 0.15).astype(int)
    df["diq010"] = np.where(df["t2d"].eq(1) & (rng.random(n) < 0.7), 1, 2)
    df["t2d_undiag"] = ((df["t2d"] == 1) & (df["diq010"] != 1)).astype(int)
    return df


# ---------- заборонені ознаки ----------
def test_features_not_forbidden() -> None:
    pats = [p.lower() for p in FORBIDDEN_FEATURE_PATTERNS]
    bad = [f for f in FEATURES if any(p in f.lower() for p in pats)]
    assert bad == []
    assert_no_forbidden_features(FEATURES)  # не піднімає


@pytest.mark.parametrize("col", ["LBXGH", "LBXGLU", "DIQ010", "DIQ160", "hba1c", "insulin"])
def test_forbidden_columns_rejected(col: str) -> None:
    with pytest.raises(ForbiddenFeatureError):
        assert_no_forbidden_features([*FEATURES, col])


def test_split_rejects_forbidden_feature(synth: pd.DataFrame) -> None:
    with pytest.raises(ForbiddenFeatureError):
        split(synth.assign(LBXGH=5.0), features=[*FEATURES, "LBXGH"])


def test_metrics_features_match_config(metrics: dict) -> None:
    assert metrics["features"] == FEATURES
    assert_no_forbidden_features(metrics["features"])


def test_nhanes_insulin_code_rejected() -> None:
    with pytest.raises(ForbiddenFeatureError):
        assert_no_forbidden_features(["LBXIN"])


# ---------- розбиття і препроцесинг ----------
def test_split_disjoint_and_stratified(synth: pd.DataFrame) -> None:
    X_tr, X_te, y_tr, y_te = split(synth)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(synth)
    assert X_tr.index.equals(y_tr.index) and X_te.index.equals(y_te.index)
    assert abs(y_tr.mean() - y_te.mean()) <= 0.01
    assert abs(y_tr.mean() - synth["t2d"].mean()) <= 0.01


def test_preprocessing_inside_pipeline() -> None:
    # Імпутер/скейлер — кроки Pipeline (fit лише на тренувальних фолдах), модель — останній крок;
    # у logit і rf є імпутер, у logit — скейлер; hgb обробляє NaN нативно
    steps = {name: make().named_steps for name, make in MAKERS.items()}
    assert list(MAKERS) == MODELS
    assert "impute" in steps["logit"] and "scale" in steps["logit"]
    assert "impute" in steps["rf"]
    for name, make in MAKERS.items():
        assert list(make().named_steps)[-1] == "model", name


# ---------- поріг і вибір моделі ----------
def test_thresholds_not_default(metrics: dict) -> None:
    # Поріг — Юден на OOF, а не 0.5; лежить у (0, 1)
    assert set(metrics["thresholds"]) == set(MODELS)
    for name, t in metrics["thresholds"].items():
        assert t != 0.5 and 0 < t < 1, name
        assert metrics["test"][name]["threshold"] == t, name


def test_best_model_by_cv_auc(metrics: dict) -> None:
    assert metrics["best_model_rule"] == "cv_auc_mean"
    cv = {m: metrics["cv"][m]["cv_auc_mean"] for m in MODELS}
    assert metrics["best_model"] == max(cv, key=cv.get)


def test_best_model_rule_in_code_uses_cv_not_test(synth: pd.DataFrame) -> None:
    # evaluate_on_test: CV-найкраща (rf) обирається, навіть коли за test AUC вона остання
    X_tr, X_te, y_tr, y_te = split(synth)
    fits = [MAKERS["logit"]().fit(X_tr, y_tr), MAKERS["logit"]().fit(X_tr, 1 - y_tr)]
    good, bad = sorted(fits, key=lambda p: -roc_auc_score(y_te, p.predict_proba(X_te)[:, 1]))
    models = {"logit": good, "rf": bad, "hgb": good}
    cv = {n: {"cv_auc_mean": 0.9 if n == "rf" else 0.6, "best_params": {}} for n in models}
    result = TrainResult(models, dict.fromkeys(models, 0.2), cv, {}, X_te.index, X_tr.index)
    m = evaluate_on_test(result, synth)
    assert m["test_auc_rank"][0] != "rf"  # передумова: за тестом rf не найкраща
    assert m["best_model"] == "rf"


# ---------- FINDRISC (c) з витоком ----------
def test_findrisc_c_invalid_in_metrics(metrics: dict) -> None:
    # Варіант (c) містить цільову інформацію: valid=False і не порівнюється з моделлю
    assert metrics["findrisc"]["c"]["valid"] is False
    assert metrics["findrisc"]["a"]["valid"] and metrics["findrisc"]["b"]["valid"]
    assert "c" not in {r["variant"] for r in metrics["delong_vs_findrisc"]}


def test_findrisc_c_never_compared_with_model(synth: pd.DataFrame) -> None:
    # Навіть коли (c) має найвищий AUC, з моделлю порівнюються лише (a) і (b)
    v = variants(synth)
    assert not v["c"].is_valid_for_comparison
    assert roc_auc_score(v["c"].y, v["c"].score) > roc_auc_score(v["a"].y, v["a"].score)
    X_tr, X_te, y_tr, _ = split(synth)
    model = MAKERS["logit"]().fit(X_tr, y_tr)
    models = dict.fromkeys(MODELS, model)
    cv = {n: {"cv_auc_mean": 0.7, "best_params": {}} for n in MODELS}
    result = TrainResult(models, dict.fromkeys(MODELS, 0.2), cv, {}, X_te.index, X_tr.index)
    m = evaluate_on_test(result, synth)
    assert [r["variant"] for r in m["delong_vs_findrisc"]] == ["a", "b"]
    assert "c" not in set(findrisc_vs_model_table(m)["Варіант FINDRISC"])


# ---------- дисклеймер ----------
def test_disclaimer_text() -> None:
    assert "Результат є дослідницькою оцінкою моделі, а не медичним діагнозом." in DISCLAIMER
