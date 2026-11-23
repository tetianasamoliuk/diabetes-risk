"""Реальна когорта: розмір, поширеність, частка пропусків у ознаках (slow)."""

import pandas as pd
import pytest

from risk.config import FEATURES, PROCESSED_DIR
from risk.data.weights import prevalence

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "cohort.parquet")


def test_size_and_positives(cohort: pd.DataFrame) -> None:
    assert 12_500 <= len(cohort) <= 13_400
    assert 2_200 <= cohort["t2d"].sum() <= 2_600


def test_weighted_prevalence(cohort: pd.DataFrame) -> None:
    assert 12 <= prevalence(cohort)["weighted_pct"][0] <= 16


def test_share_diagnosed_among_positives(cohort: pd.DataFrame) -> None:
    pos = cohort[cohort["t2d"] == 1]
    assert 0.75 <= (pos["diq010"] == 1).mean() <= 0.90


def test_features_missing_share(cohort: pd.DataFrame) -> None:
    assert (cohort[FEATURES].isna().mean() <= 0.15).all()
