"""Тест сторінки 4 (моделі): збережені моделі на користувацькому наборі."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from risk.config import FEATURES, PROCESSED_DIR

PAGE = Path(__file__).resolve().parents[2] / "app" / "pages" / "4_🧮_Моделі.py"


def run(user_df: pd.DataFrame) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=120)
    at.session_state["user_df"] = user_df
    return at.run()


def _cohort() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "cohort.parquet").head(300)


def test_user_data_shows_three_models() -> None:
    at = run(_cohort())
    assert at.exception == []
    tables = [d.value for d in at.dataframe if "Позитивних" in d.value.columns]
    assert len(tables) == 1 and len(tables[0]) == 3


def test_missing_feature_shows_info() -> None:
    at = run(_cohort().drop(columns=[FEATURES[0]]))
    assert at.exception == []
    assert any(FEATURES[0] in i.value for i in at.info)
