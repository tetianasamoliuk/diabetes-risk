"""Зважені оцінки поширеності зі складного дизайну NHANES (лінеаризація Тейлора)."""

import pandas as pd

from risk.config import Z95


def survey_mean_se(
    y: pd.Series, w: pd.Series, strata: pd.Series, psu: pd.Series
) -> tuple[float, float]:
    """Зважене середнє та SE (with-replacement PSU, як у R survey за замовчуванням)."""
    p = (y * w).sum() / w.sum()
    psu_tot = (w * (y - p) / w.sum()).groupby([strata, psu]).sum()
    var = sum(
        len(z) / (len(z) - 1) * ((z - z.mean()) ** 2).sum() for _, z in psu_tot.groupby(level=0)
    )
    return float(p), float(var**0.5)


def _design_data(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = df.dropna(subset=[outcome])
    return data[data["wt_mec"] > 0]


def _row(label: str, g: pd.DataFrame, outcome: str) -> dict:
    p, se = survey_mean_se(g[outcome], g["wt_mec"], g["strata"], g["psu"])
    return {
        "group": label,
        "n": len(g),
        "unweighted_pct": g[outcome].mean() * 100,
        "weighted_pct": p * 100,
        "ci_low": (p - Z95 * se) * 100,
        "ci_high": (p + Z95 * se) * 100,
    }


def prevalence(df: pd.DataFrame, outcome: str = "t2d", by: str | None = None) -> pd.DataFrame:
    """Поширеність (%) з 95% CI; by=None -> один рядок «Разом», інакше рядок на групу."""
    data = _design_data(df, outcome)
    if by is None:
        return pd.DataFrame([_row("Разом", data, outcome)])
    return pd.DataFrame([_row(str(k), g, outcome) for k, g in data.groupby(by)])


def design_effect(df: pd.DataFrame, outcome: str = "t2d") -> tuple[float, float]:
    """Дизайн-ефект зваженої поширеності і ефективний обсяг вибірки: (deff, n_eff).

    deff = Var_дизайн / Var_ПВВ, де Var_ПВВ = p(1 − p)/n — дисперсія за простої випадкової
    вибірки того самого обсягу; n_eff = n / deff.
    """
    data = _design_data(df, outcome)
    p, se = survey_mean_se(data[outcome], data["wt_mec"], data["strata"], data["psu"])
    deff = se**2 / (p * (1 - p) / len(data))
    return float(deff), float(len(data) / deff)
