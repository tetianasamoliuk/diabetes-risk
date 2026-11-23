"""Тести фабрик пайплайнів, сіток гіперпараметрів і запобіжника ознак."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from risk.config import FEATURES, SEED
from risk.errors import ForbiddenFeatureError
from risk.models.pipelines import (
    MAKERS,
    PARAM_GRIDS,
    assert_no_forbidden_features,
    cv_splitter,
    make_hgb,
    make_logit,
    make_rf,
    monotonic_cst,
)


def _xy_with_nan(n: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = (X["age"] + rng.normal(size=n) > 0).astype(int).to_numpy()
    X.loc[::7, "sbp"] = np.nan
    return X, y


# --- запобіжник ознак ---


@pytest.mark.parametrize(
    "columns", [["age", "LBXGH"], ["hba1c"], ["DIQ010"], ["t2d"], ["lbxglu"], ["insulin"]]
)
def test_forbidden_features_raise(columns: list[str]) -> None:
    with pytest.raises(ForbiddenFeatureError):
        assert_no_forbidden_features(columns)


def test_forbidden_error_names_column() -> None:
    with pytest.raises(ForbiddenFeatureError, match="t2d_undiag"):
        assert_no_forbidden_features(["age", "t2d_undiag"])


def test_model_features_pass() -> None:
    assert_no_forbidden_features(FEATURES)


# --- препроцесинг усередині Pipeline ---


@pytest.mark.parametrize("make", [make_logit, make_rf])
def test_imputer_inside_pipeline_before_model(make: Callable[[], Pipeline]) -> None:
    steps = [step for _, step in make().steps]
    assert isinstance(steps[0], SimpleImputer)
    assert steps[0].strategy == "median"
    assert len(steps) >= 2  # імпутер — не останній крок, модель після нього


def test_logit_scales_before_model() -> None:
    assert [name for name, _ in make_logit().steps] == ["impute", "scale", "model"]


def test_hgb_single_step_handles_nan() -> None:
    pipe = make_hgb()
    assert len(pipe.steps) == 1
    assert isinstance(pipe[-1], HistGradientBoostingClassifier)
    X, y = _xy_with_nan()
    proba = pipe.set_params(model__max_iter=30).fit(X, y).predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("make", [make_logit, make_rf])
def test_imputed_pipelines_fit_with_nan(make: Callable[[], Pipeline]) -> None:
    X, y = _xy_with_nan()
    assert np.isfinite(make().fit(X, y).predict_proba(X)).all()


def test_seed_everywhere() -> None:
    assert make_logit()[-1].random_state == SEED
    assert make_rf()[-1].random_state == SEED
    assert make_hgb()[-1].random_state == SEED


def test_hgb_without_implicit_early_stopping() -> None:
    """early_stopping="auto" вмикається лише при n > 10 000: CV-фолди і refit були б різні."""
    assert make_hgb()[-1].early_stopping is False


# --- сітки гіперпараметрів ---


def test_param_grids_are_fixed() -> None:
    assert PARAM_GRIDS["logit"] == {"model__C": [0.1, 1, 10]}
    assert PARAM_GRIDS["rf"] == {"model__max_depth": [None, 8], "model__min_samples_leaf": [5, 20]}
    assert PARAM_GRIDS["hgb"] == {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [15, 31],
    }
    assert make_rf()[-1].n_estimators == 300
    assert make_hgb()[-1].max_iter == 200


@pytest.mark.parametrize("name", list(MAKERS))
def test_param_grids_apply_to_pipelines(name: str) -> None:
    for params in ParameterGrid(PARAM_GRIDS[name]):
        MAKERS[name]().set_params(**params)  # невідома назва параметра → ValueError


# --- монотонні обмеження ---


def test_monotonic_cst_follows_features_order() -> None:
    cst = monotonic_cst()
    assert len(cst) == len(FEATURES)
    assert {f: c for f, c in zip(FEATURES, cst, strict=True) if c} == {
        "bmi": 1,
        "waist": 1,
        "pa_h_week": -1,
    }
    assert make_rf()[-1].monotonic_cst == cst and make_hgb()[-1].monotonic_cst == cst


def _u_shaped_xy(n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    """Ризик U-подібний за ІМТ, талією й активністю: без обмежень дерева були б немонотонні."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    z = X["age"] + X["bmi"] ** 2 + X["waist"] ** 2 + X["pa_h_week"] ** 2 - 2
    return X, (z + rng.normal(size=n) > 0).astype(int).to_numpy()


@pytest.mark.parametrize(
    "pipe",
    [make_rf().set_params(model__n_estimators=30), make_hgb().set_params(model__max_iter=30)],
    ids=["rf", "hgb"],
)
def test_scenario_changes_do_not_raise_proba(pipe: Pipeline) -> None:
    X, y = _u_shaped_xy()
    pipe.fit(X, y)
    base = pipe.predict_proba(X)[:, 1]
    for col, step in (("bmi", -1.0), ("waist", -1.0), ("pa_h_week", 1.0)):
        p = pipe.predict_proba(X.assign(**{col: X[col] + step}))[:, 1]
        assert (p <= base + 1e-9).all(), col  # допуск — порядок підсумовування дерев RF


def test_cv_splitter() -> None:
    cv = cv_splitter()
    assert isinstance(cv, StratifiedKFold)
    assert (cv.n_splits, cv.shuffle, cv.random_state) == (5, True, SEED)
