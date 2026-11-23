"""Гармонізація змінних NHANES: тиск, активність, ліки, цільова змінна."""

import numpy as np
import pandas as pd
import pytest

from risk.config import DBP_COLS, SBP_COLS
from risk.data import harmonize as h

NAN = np.nan


def frame(**cols: list) -> pd.DataFrame:
    return pd.DataFrame(cols)


@pytest.mark.parametrize("cycle", ["2015-2016", "2017-2020"])
def test_bp_both_cycles(cycle: str) -> None:
    sbp, dbp = SBP_COLS[cycle], DBP_COLS[cycle]
    raw = pd.DataFrame({c: [120.0, NAN, 100.0] for c in sbp} | {c: [80.0, NAN, 0.0] for c in dbp})
    raw[sbp[0]] = [110.0, NAN, 100.0]
    raw[dbp[1]] = [70.0, NAN, 0.0]
    s = h.mean_bp(raw, sbp, drop_zero=False)
    d = h.mean_bp(raw, dbp, drop_zero=True)
    n = len(sbp)
    assert s[0] == pytest.approx((110 + 120 * (n - 1)) / n)
    assert np.isnan(s[1])
    assert d[0] == pytest.approx((80 * (len(dbp) - 1) + 70) / len(dbp))
    assert np.isnan(d[1])
    assert np.isnan(d[2])  # усі нулі -> NaN


def test_bp_partial_mean() -> None:
    raw = frame(a=[100.0], b=[NAN], c=[120.0])
    assert h.mean_bp(raw, ["a", "b", "c"], drop_zero=True)[0] == 110.0


def test_yes_no() -> None:
    out = h.yes_no(pd.Series([1, 2, 7, 9, NAN]))
    assert out.iloc[:2].tolist() == [1.0, 0.0]
    assert out.iloc[2:].isna().all()


def pa_frame(**over: list) -> pd.DataFrame:
    cols = {c: [NAN] for d in h.PA_DOMAINS for c in d}
    for did, _, _ in h.PA_DOMAINS:
        cols[did] = [2]
    return pd.DataFrame(cols | over)


def test_pa_all_no_is_zero() -> None:
    assert h.pa_minutes_week(pa_frame())[0] == 0.0


def test_pa_sum_and_codes() -> None:
    raw = pa_frame(PAQ605=[1], PAQ610=[3], PAD615=[60], PAQ620=[1], PAQ625=[2], PAD630=[30])
    assert h.pa_minutes_week(raw)[0] == 3 * 60 + 2 * 30


@pytest.mark.parametrize(("days", "mins"), [(77, 60), (99, 60), (3, 9999), (3, 7777), (3, NAN)])
def test_pa_invalid_codes_nan(days: float, mins: float) -> None:
    raw = pa_frame(PAQ605=[1], PAQ610=[days], PAD615=[mins])
    assert np.isnan(h.pa_minutes_week(raw)[0])


def test_pa_did_missing_nan() -> None:
    assert np.isnan(h.pa_minutes_week(pa_frame(PAQ605=[9]))[0])


def test_bp_medication() -> None:
    raw = frame(
        BPQ050A=[1, 2, NAN, NAN, NAN, NAN, 1],
        BPQ020=[1, 1, 2, 1, 1, 9, 2],
        BPQ040A=[1, 1, NAN, 2, 1, NAN, 2],
    )
    out = h.bp_medication(raw)
    assert out[:4].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(out[4]) and np.isnan(out[5])
    assert out[6] == 1.0  # явна відповідь не перекривається


def test_cholesterol_medication() -> None:
    raw = frame(
        BPQ100D=[1, 2, NAN, NAN, NAN, NAN, 1],
        BPQ090D=[1, 1, 2, NAN, NAN, 9, 1],
        BPQ080=[1, 1, 1, 2, 9, 1, 2],
    )
    out = h.cholesterol_medication(raw)
    assert out[:4].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.isnan(out[4]) and np.isnan(out[5])
    assert out[6] == 1.0  # явна відповідь не перекривається


def test_t1d_proxy() -> None:
    raw = frame(
        DIQ010=[1, 1, 1, 1, 1, 2, 3],
        DID040=[25, 25, 45, 666, 999, 20, NAN],
        DIQ050=[1, 2, 1, 1, 1, 1, NAN],
    )
    out = h.t1d_proxy(raw)
    assert out[:5].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]  # 666 — «менше 1 року»
    assert out[5:].isna().all()  # недіагностованим NaN


def test_high_glucose_history() -> None:
    raw = frame(
        DIQ010=[2, 2, 3, 1, 2, 2, 3],
        DIQ160=[1, 2, NAN, NAN, 7, NAN, 2],
    )
    out = h.high_glucose_history(raw)
    assert out[:3].tolist() == [1.0, 0.0, 1.0]
    assert out[4:6].isna().all() and np.isnan(out[3])
    assert out[6] == 1.0


@pytest.mark.parametrize(
    ("hba1c", "diq", "expected"),
    [
        (6.49, 2, 0.0),
        (6.5, 2, 1.0),
        (6.51, 2, 1.0),
        (5.0, 1, 1.0),
        (5.0, 3, 0.0),
        (5.0, 7, NAN),
        (5.0, 9, NAN),
        (7.0, 9, 1.0),
        (NAN, 1, NAN),
        (NAN, 2, NAN),
    ],
)
def test_target(hba1c: float, diq: float, expected: float) -> None:
    out = h.target(frame(LBXGH=[hba1c], DIQ010=[diq]))[0]
    assert (np.isnan(out) and np.isnan(expected)) or out == expected


def test_target_undiagnosed() -> None:
    raw = frame(LBXGH=[6.5, 6.4, 8.0, NAN], DIQ010=[2, 3, 1, 2])
    out = h.target_undiagnosed(raw)
    assert out[:2].tolist() == [1.0, 0.0]
    assert out[2:].isna().all()


def full_raw() -> pd.DataFrame:
    cols = {c: [1.0, 2.0] for c in SBP_COLS["2017-2020"] + DBP_COLS["2017-2020"]}
    for d in h.PA_DOMAINS:
        cols |= {d[0]: [2, 2], d[1]: [NAN, NAN], d[2]: [NAN, NAN]}
    return pd.DataFrame(
        cols
        | dict(
            SEQN=[1, 2], RIDAGEYR=[30, 50], RIAGENDR=[1, 2], BMXBMI=[25.0, 30.0],
            BMXWAIST=[90.0, 80.0], LBXTC=[180.0, 200.0], LBDHDD=[50.0, 60.0],
            MCQ300C=[1, 2], BPQ050A=[NAN, NAN], BPQ020=[2, 1], BPQ040A=[NAN, 2],
            DIQ010=[2, 1], DIQ160=[1, NAN], LBXGH=[5.0, 7.0], WTMECPRP=[520.0, 0.0],
            BPQ100D=[NAN, 1], BPQ090D=[NAN, 1], BPQ080=[2, 1], DID040=[NAN, 50], DIQ050=[NAN, 2],
            SDMVPSU=[1, 2], SDMVSTRA=[10, 11], RIDSTATR=[2, 2], RIDEXPRG=[2.0, NAN],
        )
    )  # fmt: skip


def test_harmonize_schema_and_weights() -> None:
    out = h.harmonize(full_raw(), "2017-2020")
    expected = (
        "SEQN cycle cycle_2017 age male bmi waist sbp dbp tc hdl pa_min_week pa_h_week fam_dm "
        "bp_med high_glu_hist chol_med t1d_proxy diq010 hba1c t2d t2d_undiag wt_mec psu strata "
        "examined pregnant"
    ).split()
    assert set(expected) <= set(out.columns)
    assert (out["cycle_2017"] == 1.0).all()
    assert out["wt_mec"][0] == pytest.approx(520 * 3.2 / 5.2)
    assert out["male"].tolist() == [1.0, 0.0]
    assert out["t2d"].tolist() == [0.0, 1.0]
    assert out["pa_h_week"].tolist() == [0.0, 0.0]
    assert out["pregnant"].tolist() == [False, False]
    assert out["chol_med"].tolist() == [0.0, 1.0]
    assert np.isnan(out["t1d_proxy"][0]) and out["t1d_proxy"][1] == 0.0
