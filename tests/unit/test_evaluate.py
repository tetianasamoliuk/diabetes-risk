"""Тести порогу Youden, метрик класифікації та калібрування."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.proportion import proportion_confint

from risk.models.calibration import (
    brier,
    calibration_slope_intercept,
    calibration_summary,
    calibration_table,
)
from risk.models.delong import auc_ci
from risk.models.evaluate import (
    Metrics,
    classification_metrics,
    ppv_at_prevalence,
    roc_points,
    wilson_ci,
)
from risk.models.threshold import youden_threshold


def _random_data(seed: int, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, n)
    p = np.clip(0.15 + 0.3 * y + rng.normal(0, 0.2, n), 0, 1).round(2)  # округлення → зв'язки
    return y, p


# --- threshold ---


def test_youden_perfect_separation_between_classes() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 200)
    p = np.where(y == 1, 0.6 + 0.4 * rng.random(200), 0.5 * rng.random(200))
    thr = youden_threshold(y, p)
    assert p[y == 0].max() < thr <= p[y == 1].min()


def test_youden_hand_example_tie_takes_smallest() -> None:
    """Пороги ↓: 0.8 J=1/3; 0.6 J=2/3−0; 0.4 J=1/3; 0.35 J=1−1/3; 0.3 J=1/3; 0.1 J=0.

    Максимум 2/3 двічі (0.6 і 0.35; у float 0.666…66 ≠ 0.666…67) → найменший поріг 0.35.
    """
    y = [0, 0, 1, 0, 1, 1]
    p = [0.1, 0.3, 0.35, 0.4, 0.6, 0.8]
    assert youden_threshold(y, p) == 0.35


def test_youden_hand_example_unique_max() -> None:
    # 0.7: J=1/2; 0.5: tpr=1, fpr=1/3 (негатив 0.5 теж ≥ поріг) → J=2/3 — максимум; 0.4: 1/3; 0.2: 0
    y = [0, 1, 0, 1, 0]
    p = [0.2, 0.5, 0.4, 0.7, 0.5]
    assert youden_threshold(y, p) == 0.5


def test_youden_is_finite_and_maximizes_j() -> None:
    y, p = _random_data(1)
    thr = youden_threshold(pd.Series(y), pd.Series(p))
    assert np.isfinite(thr)
    fpr, tpr, _ = roc_curve(y, p)
    pred = p >= thr
    j = pred[y == 1].mean() - pred[y == 0].mean()
    assert j == pytest.approx((tpr - fpr).max())


# --- evaluate ---


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threshold", [0.1, 0.25, 0.4, 0.6])
def test_classification_metrics_match_sklearn(seed: int, threshold: float) -> None:
    y, p = _random_data(seed)
    m = classification_metrics(y, p, threshold)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    assert (m.tn, m.fp, m.fn, m.tp) == (tn, fp, fn, tp)
    assert m.accuracy == pytest.approx(accuracy_score(y, pred))
    assert m.precision == pytest.approx(precision_score(y, pred, zero_division=0))
    assert m.recall == pytest.approx(recall_score(y, pred))
    assert m.f1 == pytest.approx(f1_score(y, pred, zero_division=0))
    neg = y == 0
    assert m.specificity == pytest.approx(np.sum((pred == 0) & neg) / np.sum(neg))
    assert m.auc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert (m.auc, m.auc_ci_low, m.auc_ci_high) == auc_ci(y, p)
    assert m.brier == pytest.approx(brier_score_loss(y, p))
    assert m.threshold == threshold
    assert m.n == len(y)
    assert (m.se_ci_low, m.se_ci_high) == wilson_ci(tp, tp + fn)
    assert (m.sp_ci_low, m.sp_ci_high) == wilson_ci(tn, tn + fp)
    assert m.se_ci_low <= m.recall <= m.se_ci_high
    assert (m.calib_slope, m.calib_intercept) == calibration_slope_intercept(y, p)


def test_threshold_inclusive() -> None:
    y = np.array([1, 1, 0, 0])
    m = classification_metrics(y, np.array([0.5, 0.9, 0.5, 0.1]), 0.5)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 0, 1)


def test_precision_zero_when_no_positive_predictions() -> None:
    y, p = _random_data(3)
    m = classification_metrics(y, p, threshold=1.01)
    assert m.tp == m.fp == 0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.specificity == 1.0
    assert np.isnan(m.ppv_ci_low) and np.isnan(m.ppv_ci_high)  # PPV без позитивних прогнозів


def test_wilson_ci_by_hand() -> None:
    # 392 з 496: p̂ = 0.790; z = 1.96 → центр (p̂ + z²/2n)/(1 + z²/n), півширина z√(…)/(1 + z²/n)
    low, high = wilson_ci(392, 496)
    z, n, ph = 1.959964, 496, 392 / 496
    centre = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    assert (low, high) == pytest.approx((centre - half, centre + half), abs=1e-6)
    assert (low, high) == pytest.approx(proportion_confint(392, 496, method="wilson"))
    assert all(np.isnan(wilson_ci(0, 0)))


def test_ppv_at_prevalence() -> None:
    # Se 0.8, Sp 0.9, π 0.1: 0.08 / (0.08 + 0.09)
    assert ppv_at_prevalence(0.8, 0.9, 0.1) == pytest.approx(0.08 / 0.17)
    assert ppv_at_prevalence(0.8, 0.9, 0.3) > ppv_at_prevalence(0.8, 0.9, 0.1)
    assert ppv_at_prevalence(0.8, 1.0, 0.1) == 1.0


def test_calibration_slope_intercept_perfect() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 100_000)
    y = rng.binomial(1, p)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.03) and intercept == pytest.approx(0, abs=0.03)


def test_calibration_slope_detects_overconfidence_and_shift() -> None:
    rng = np.random.default_rng(1)
    lp = rng.normal(-1.5, 1.2, 100_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    sharp = 1 / (1 + np.exp(-2 * lp))  # надто впевнені прогнози: logit удвічі розтягнутий
    low = 1 / (1 + np.exp(-(lp - 0.5)))  # систематично занижений ризик
    assert calibration_slope_intercept(y, sharp)[0] == pytest.approx(0.5, abs=0.03)
    assert calibration_slope_intercept(y, low)[1] == pytest.approx(0.5, abs=0.03)


def test_metrics_is_frozen() -> None:
    y, p = _random_data(5)
    m = classification_metrics(y, p, 0.3)
    assert isinstance(m, Metrics)
    with pytest.raises(AttributeError):
        m.auc = 0.5


def test_roc_points_match_sklearn() -> None:
    y, p = _random_data(6)
    fpr, tpr = roc_points(pd.Series(y), pd.Series(p))
    ref_fpr, ref_tpr, _ = roc_curve(y, p)
    np.testing.assert_array_equal(fpr, ref_fpr)
    np.testing.assert_array_equal(tpr, ref_tpr)


# --- calibration ---


def test_calibration_perfectly_calibrated() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 50_000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p)
    assert list(table.columns) == ["bin", "mean_pred", "frac_pos", "n"]
    assert len(table) == 10
    assert table["n"].sum() == len(y)
    # SE частки в біні ≈ √(0.25/5000) ≈ 0.007 → допуск ~4 SE
    assert (table["mean_pred"] - table["frac_pos"]).abs().max() < 0.03


def test_calibration_table_values_by_hand() -> None:
    y = [0, 0, 1, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    table = calibration_table(y, p, n_bins=2)
    assert table["bin"].tolist() == [1, 2]
    assert table["n"].tolist() == [3, 3]
    np.testing.assert_allclose(table["mean_pred"], [0.2, 0.7])
    np.testing.assert_allclose(table["frac_pos"], [1 / 3, 2 / 3])


def test_calibration_table_duplicate_edges_dropped() -> None:
    rng = np.random.default_rng(1)
    p = rng.choice([0.1, 0.5, 0.9], 1000)
    y = rng.binomial(1, p)
    table = calibration_table(pd.Series(y), pd.Series(p))
    assert len(table) < 10
    assert table["n"].sum() == 1000


def test_brier_and_summary() -> None:
    y, p = _random_data(7)
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))
    mean_pred, mean_obs = calibration_summary(pd.Series(y), pd.Series(p))
    assert mean_pred == pytest.approx(p.mean())
    assert mean_obs == pytest.approx(y.mean())
