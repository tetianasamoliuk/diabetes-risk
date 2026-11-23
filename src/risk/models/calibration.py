"""Калібрування ймовірностей: квантильні біни, Brier, нахил і зсув калібрування."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss

from risk.config import CALIBRATION_BINS

P_CLIP = 1e-6  # прогнози 0 і 1 обрізаються, щоб logit(p) був скінченним


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Квантильні біни прогнозу: bin (з 1), mean_pred, frac_pos (частка y == 1), n.

    Однакові межі квантилів (багато зв'язок у p) зливаються — бінів може бути менше за n_bins.
    """
    df = pd.DataFrame({"y": np.asarray(y).ravel() == 1, "p": np.asarray(p, dtype=float).ravel()})
    df["bin"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop") + 1
    return (
        df.groupby("bin")
        .agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
        .reset_index()
    )


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier = mean((p − y)²); 0 — ідеально, 0.25 — константа 0.5."""
    return float(brier_score_loss(np.asarray(y).ravel() == 1, np.asarray(p, dtype=float).ravel()))


def calibration_summary(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(середній прогноз, спостережувана частка y == 1) — калібрування «в середньому»."""
    return float(np.mean(np.asarray(p, dtype=float))), float(np.mean(np.asarray(y) == 1))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Нахил і зсув калібрування (логістична регресія без регуляризації).

    Нахил — коефіцієнт β₁ у logit P(y = 1) = β₀ + β₁·logit(p): 1 — ідеально, < 1 — прогнози
    надто впевнені. Зсув (calibration-in-the-large) — β₀ при фіксованому β₁ = 1 (logit(p) як
    offset), у log-odds: 0 — ідеально, > 0 — модель у середньому занижує ризик.
    """
    yy = (np.asarray(y).ravel() == 1).astype(float)
    lp = np.asarray(p, dtype=float).ravel().clip(P_CLIP, 1 - P_CLIP)
    lp = np.log(lp / (1 - lp))
    slope = sm.Logit(yy, sm.add_constant(lp)).fit(disp=0).params[1]
    ones = np.ones_like(lp)
    intercept = sm.GLM(yy, ones, family=sm.families.Binomial(), offset=lp).fit().params[0]
    return float(slope), float(intercept)
