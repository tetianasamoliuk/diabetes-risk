"""Читання XPT і склеювання модулів циклу NHANES."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk.config import RAW_DIR
from risk.data import load
from risk.data.load import load_cycle, normalize_xpt_zeros, read_xpt


def test_normalize_zeros() -> None:
    df = pd.DataFrame({"a": [5.4e-79, 1e-5, np.nan, -5.4e-79, 3.0], "s": list("abcde")})
    out = normalize_xpt_zeros(df)
    assert out["a"].iloc[0] == 0.0
    assert out["a"].iloc[3] == 0.0
    assert out["a"].iloc[1] == 1e-5
    assert np.isnan(out["a"].iloc[2])
    assert out["a"].iloc[4] == 3.0
    assert list(out["s"]) == list("abcde")
    assert df["a"].iloc[0] == 5.4e-79  # вхідний df не мутує


def test_load_cycle_joins_and_drops_duplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_read(path: Path) -> pd.DataFrame:
        base = {"SEQN": [1, 2, 3]}
        if path.stem == "DEMO_I":
            return pd.DataFrame(base | {"RIDAGEYR": [30, 40, 50]})
        if path.stem == "GHB_I":
            return pd.DataFrame({"SEQN": [1, 3], "LBXGH": [5.0, 7.0], "RIDAGEYR": [99, 99]})
        return pd.DataFrame({"SEQN": [2]})

    monkeypatch.setattr(load, "read_xpt", fake_read)
    df = load_cycle("2015-2016", tmp_path)
    assert len(df) == 3
    assert list(df["RIDAGEYR"]) == [30, 40, 50]
    assert df["LBXGH"].isna().tolist() == [False, True, False]
    assert (df["cycle"] == "2015-2016").all()


@pytest.mark.slow
@pytest.mark.skipif(not (RAW_DIR / "GHB_I.xpt").exists(), reason="data/raw не завантажено")
def test_read_xpt_real() -> None:
    df = read_xpt(RAW_DIR / "GHB_I.xpt")
    assert df["SEQN"].dtype == "int64"
    assert "LBXGH" in df
    assert (
        not ((df.select_dtypes("number").abs() < 1e-70) & (df.select_dtypes("number") != 0))
        .any()
        .any()
    )


def test_read_xpt_normalizes_and_casts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = pd.DataFrame({"SEQN": [1.0, 2.0], "x": [5.4e-79, 2.0]})
    seen: dict[str, object] = {}

    def fake_sas(path: Path, format: str) -> pd.DataFrame:
        seen.update(path=path, format=format)
        return raw

    monkeypatch.setattr(load.pd, "read_sas", fake_sas)
    out = read_xpt(tmp_path / "A.xpt")
    assert seen == {"path": tmp_path / "A.xpt", "format": "xport"}
    assert out["SEQN"].dtype == "int64"
    assert list(out["x"]) == [0.0, 2.0]
