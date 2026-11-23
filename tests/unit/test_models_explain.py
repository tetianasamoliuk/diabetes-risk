"""Тести SHAP-пояснень для HistGradientBoosting."""

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.pipeline import Pipeline

from risk.errors import DataValidationError
from risk.models.explain import global_shap, local_shap, make_explainer, top_contributions
from risk.models.pipelines import make_hgb, make_logit


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["age", "bmi", "waist", "hdl"])
    y = (X["age"] + 0.5 * X["bmi"] + rng.normal(size=300) > 0.3).astype(int).to_numpy()
    X.loc[rng.random(300) < 0.2, "bmi"] = np.nan  # пояснення мають працювати на сирих NaN
    X.loc[rng.random(300) < 0.1, "age"] = np.nan
    return X, y


@pytest.fixture(scope="module")
def pipe(data: tuple[pd.DataFrame, np.ndarray]) -> Pipeline:
    X, y = data
    return make_hgb().set_params(model__max_iter=30, model__monotonic_cst=None).fit(X, y)


def test_global_shap_additive_with_nan(
    pipe: Pipeline, data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X, _ = data
    expl = global_shap(make_explainer(pipe), X)
    assert isinstance(expl, shap.Explanation)
    assert expl.values.shape == X.shape
    assert expl.feature_names == list(X.columns)
    total = np.asarray(expl.base_values) + expl.values.sum(axis=1)
    np.testing.assert_allclose(total, pipe.decision_function(X), atol=1e-6)  # log-odds


def test_local_shap_one_row(pipe: Pipeline, data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, _ = data
    row = X.iloc[[5]]
    expl = local_shap(make_explainer(pipe), row)
    assert expl.values.shape == (X.shape[1],)
    assert np.ndim(expl.base_values) == 0
    total = float(expl.base_values) + expl.values.sum()
    assert total == pytest.approx(pipe.decision_function(row)[0], abs=1e-6)


def test_local_shap_rejects_many_rows(
    pipe: Pipeline, data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    with pytest.raises(DataValidationError):
        local_shap(make_explainer(pipe), data[0].iloc[:2])


@pytest.mark.parametrize("k", [1, 3, 4])
def test_top_contributions_sorted_by_abs(
    pipe: Pipeline, data: tuple[pd.DataFrame, np.ndarray], k: int
) -> None:
    expl = local_shap(make_explainer(pipe), data[0].iloc[[0]])
    top = top_contributions(expl, k=k)
    assert len(top) == k
    assert [abs(v) for _, v in top] == sorted((abs(v) for _, v in top), reverse=True)
    biggest = max(abs(v) for v in expl.values)
    assert abs(top[0][1]) == pytest.approx(biggest)
    assert {f for f, _ in top} <= set(data[0].columns)


def test_top_contributions_hand_example() -> None:
    expl = shap.Explanation(
        values=np.array([0.1, -0.5, 0.3]), base_values=0.0, feature_names=["a", "b", "c"]
    )
    assert top_contributions(expl, k=2) == [("b", -0.5), ("c", 0.3)]


def test_make_explainer_rejects_imputing_pipeline(
    data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = data
    with pytest.raises(DataValidationError, match="сирих"):
        make_explainer(make_logit().fit(X, y))
