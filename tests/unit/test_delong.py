"""Тести fast DeLong (Sun & Xu 2014): AUC, коваріація, тест різниці, CI, bootstrap SE."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from risk.config import SEED
from risk.errors import DataValidationError
from risk.models.delong import (
    DelongResult,
    auc_ci,
    compute_midrank,
    delong_test,
    fast_delong,
)


def bootstrap_auc_se(
    y: np.ndarray, score: np.ndarray, n_boot: int = 500, seed: int = SEED
) -> float:
    """SE AUC стратифікованим bootstrap (окремо позитивні й негативні) — еталон для DeLong."""
    y, s = np.asarray(y), np.asarray(score, dtype=float)
    i_pos, i_neg = np.flatnonzero(y == 1), np.flatnonzero(y != 1)
    y_boot = np.r_[np.ones(i_pos.size), np.zeros(i_neg.size)]
    rng = np.random.default_rng(seed)
    draws = (
        np.r_[rng.choice(i_pos, i_pos.size), rng.choice(i_neg, i_neg.size)] for _ in range(n_boot)
    )
    aucs = [roc_auc_score(y_boot, s[idx]) for idx in draws]
    return float(np.std(aucs, ddof=1))


def _brute_cov(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Пряма O(mn) формула DeLong 1988: ψ = 1 (X > Y), ½ (X = Y), 0; S10/m + S01/n."""
    v10, v01 = [], []
    for s in scores:
        pos, neg = s[y == 1], s[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
    v10, v01 = np.array(v10), np.array(v01)
    d10 = v10 - v10.mean(axis=1, keepdims=True)
    d01 = v01 - v01.mean(axis=1, keepdims=True)
    m, n = v10.shape[1], v01.shape[1]
    cov = (d10 @ d10.T) / (m - 1) / m + (d01 @ d01.T) / (n - 1) / n
    return v10.mean(axis=1), cov


def _tied_data(seed: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Скори з великою кількістю зв'язок (цілі 0..9), слабко пов'язані з y."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    y[:2], y[2:4] = 1, 0
    scores = rng.integers(0, 10, size=(k, n)) + 2 * y
    return y, scores.astype(float)


def test_midrank_ties_hand_example() -> None:
    # відсортовано: 1→1, 2→2, три трійки на позиціях 3,4,5 → середній ранг 4
    np.testing.assert_array_equal(compute_midrank(np.array([3, 1, 3, 2, 3])), [4, 1, 4, 2, 4])


def test_midrank_no_ties_is_permutation_rank() -> None:
    x = np.array([0.5, -1.0, 2.0, 0.1])
    np.testing.assert_array_equal(compute_midrank(x), [3, 1, 4, 2])


@pytest.mark.parametrize(("seed", "n", "k"), [(0, 50, 1), (1, 300, 3), (2, 1000, 2), (3, 7, 2)])
def test_fast_delong_auc_matches_sklearn(seed: int, n: int, k: int) -> None:
    y, scores = _tied_data(seed, n, k)
    aucs, _ = fast_delong(y, scores)
    expected = [roc_auc_score(y, s) for s in scores]
    np.testing.assert_allclose(aucs, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_delong_cov_matches_bruteforce(seed: int) -> None:
    y, scores = _tied_data(seed, 200, 3)
    aucs, cov = fast_delong(y, scores)
    ref_aucs, ref_cov = _brute_cov(y, scores)
    np.testing.assert_allclose(aucs, ref_aucs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-10, atol=1e-15)


def test_fast_delong_hand_computed_components() -> None:
    """4 позитивних, 3 негативних, зі зв'язками позитив/негатив.

    a: pos [.9 .8 .4 .6], neg [.4 .3 .7]
       V10 = [3/3, 3/3, (1+½)/3, 2/3] = [24, 24, 12, 16]/24; AUC_a = 19/24
       V01 = [(3+½)/4, 4/4, 2/4]      = [21, 24, 12]/24
    b: pos [.2 .9 .5 .7], neg [.6 .1 .5]
       V10 = [1/3, 1, (1+½)/3, 1]     = [8, 24, 12, 24]/24;  AUC_b = 17/24
       V01 = [2/4, 4/4, (2+½)/4]      = [12, 24, 15]/24
    Відхилення (×24): V10_a [5 5 −7 −3], V10_b [−9 7 −5 7]; V01_a [2 5 −7], V01_b [−5 7 −2]
    S10 = [[108, 4], [4, 204]] / 576 / 3;  S01 = [[78, 39], [39, 78]] / 576 / 2
    S = S10/4 + S01/3 = [[11/288, 41/3456], [41/3456, 5/96]]
    """
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    a = [0.9, 0.8, 0.4, 0.6, 0.4, 0.3, 0.7]
    b = [0.2, 0.9, 0.5, 0.7, 0.6, 0.1, 0.5]
    aucs, cov = fast_delong(y, np.array([a, b]))
    np.testing.assert_allclose(aucs, [19 / 24, 17 / 24], rtol=0, atol=1e-15)
    expected = np.array([[11 / 288, 41 / 3456], [41 / 3456, 5 / 96]])
    np.testing.assert_allclose(cov, expected, rtol=1e-13)


def test_fast_delong_single_row_shape() -> None:
    y, scores = _tied_data(0, 100, 1)
    aucs, cov = fast_delong(y, scores[0])
    assert aucs.shape == (1,)
    assert cov.shape == (1, 1)


@pytest.mark.parametrize("seed", [11, 12])
def test_delong_se_vs_bootstrap(seed: int) -> None:
    """Відносна різниця SE DeLong і bootstrap SE < 10 % (вимога до реалізації DeLong)."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, 600)
    score = y * 1.0 + rng.normal(size=600)
    _, cov = fast_delong(y, score)
    se_delong = float(np.sqrt(cov[0, 0]))
    se_boot = bootstrap_auc_se(y, score, n_boot=1000, seed=seed)
    assert abs(se_delong - se_boot) / se_boot < 0.10


def test_bootstrap_se_deterministic_and_positive() -> None:
    y, scores = _tied_data(5, 200, 1)
    se1 = bootstrap_auc_se(y, scores[0], n_boot=200, seed=7)
    se2 = bootstrap_auc_se(y, scores[0], n_boot=200, seed=7)
    assert se1 == se2
    assert se1 > 0


def test_identical_scores_zero_diff_p_one() -> None:
    y, scores = _tied_data(0, 300, 1)
    res = delong_test(y, scores[0], scores[0])
    assert isinstance(res, DelongResult)
    assert res.diff == 0.0
    assert res.se_diff == 0.0
    assert res.z == 0.0
    assert res.p == 1.0
    assert res.ci_low == res.ci_high == 0.0


def test_monotone_transform_same_ranks_p_one() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.4, 200)
    a = rng.random(200)
    res = delong_test(y, a, np.log(a))
    assert res.diff == 0.0
    assert res.p == 1.0


def test_perfect_vs_random_significant() -> None:
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.3, 300)
    perfect = y + rng.random(300) * 0.5
    random = rng.random(300)
    res = delong_test(y, perfect, random)
    assert res.auc_a == 1.0
    assert res.p < 0.001
    assert res.z > 0
    assert res.ci_low > 0


@pytest.mark.parametrize("seed", range(5))
def test_delong_symmetry(seed: int) -> None:
    y, scores = _tied_data(seed, 400, 2)
    ab = delong_test(y, scores[0], scores[1])
    ba = delong_test(y, scores[1], scores[0])
    assert ab.z == -ba.z
    assert ab.p == ba.p
    assert ab.diff == -ba.diff
    assert ab.se_diff == ba.se_diff


def test_delong_test_fields_consistent() -> None:
    y, scores = _tied_data(9, 500, 2)
    res = delong_test(y, scores[0], scores[1])
    aucs, cov = fast_delong(y, scores)
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert res.auc_a == pytest.approx(aucs[0])
    assert res.auc_b == pytest.approx(aucs[1])
    assert res.diff == pytest.approx(aucs[0] - aucs[1])
    assert res.se_diff == pytest.approx(se)
    assert res.z == pytest.approx(res.diff / se)
    assert res.p == pytest.approx(2 * norm.sf(abs(res.z)))
    half = norm.ppf(0.975) * se
    assert (res.ci_low, res.ci_high) == pytest.approx((res.diff - half, res.diff + half))


def test_auc_ci_normal_approximation() -> None:
    y, scores = _tied_data(2, 400, 1)
    auc, low, high = auc_ci(y, scores[0])
    aucs, cov = fast_delong(y, scores)
    half = norm.ppf(0.975) * np.sqrt(cov[0, 0])
    assert auc == pytest.approx(roc_auc_score(y, scores[0]), abs=1e-12)
    assert (low, high) == pytest.approx((auc - half, auc + half))
    _, low90, high90 = auc_ci(y, scores[0], alpha=0.10)
    assert low < low90 < auc < high90 < high


def test_auc_ci_clipped_to_unit_interval() -> None:
    # один «перевернутий» парний випадок: AUC близько 1, верхня межа без обрізання > 1
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.1, 0.15, 0.05])
    auc, low, high = auc_ci(y, s)
    assert auc == pytest.approx(15 / 16)
    assert high == 1.0
    assert 0.0 <= low < auc


def test_accepts_pandas_series() -> None:
    y, scores = _tied_data(1, 120, 2)
    ys, a, b = pd.Series(y), pd.Series(scores[0]), pd.Series(scores[1])
    assert delong_test(ys, a, b) == delong_test(y, scores[0], scores[1])
    assert auc_ci(ys, a) == auc_ci(y, scores[0])


@pytest.mark.parametrize("y", [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
def test_requires_two_per_class(y: list[int]) -> None:
    with pytest.raises(DataValidationError):
        fast_delong(np.array(y), np.array([0.1, 0.2, 0.3, 0.4]))


def test_rejects_shape_mismatch_and_nan() -> None:
    y = np.array([1, 1, 0, 0])
    with pytest.raises(DataValidationError):
        fast_delong(y, np.array([0.1, 0.2, 0.3]))
    with pytest.raises(DataValidationError):
        fast_delong(y, np.array([0.1, np.nan, 0.3, 0.4]))
