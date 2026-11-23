"""Тести таблиць і текстів з metrics.json (синтетичний словник тієї ж структури)."""

import numpy as np
import pandas as pd
import pytest

from risk.errors import DataValidationError
from risk.models.summary import (
    _pair_text,
    best_model_conclusion,
    calibration_frames,
    confusion_counts,
    delong_holm,
    delong_table,
    findrisc_auc_table,
    findrisc_holm,
    findrisc_vs_model_table,
    metrics_ci_table,
    metrics_summary_table,
    pct_ci,
    subgroup,
    subgroup_auc,
    subgroup_comparisons,
    subgroup_conclusion,
    subgroup_numbers,
    subgroup_view,
)


def _test(auc: float) -> dict:
    return dict(
        accuracy=0.7, precision=0.4, recall=0.8, specificity=0.7, f1=0.5, auc=auc,
        auc_ci_low=auc - 0.02, auc_ci_high=auc + 0.02, brier=0.12,
        tn=10, fp=4, fn=2, tp=6, threshold=0.15, n=22, se_ci_low=0.752, se_ci_high=0.824,
        sp_ci_low=0.6, sp_ci_high=0.8, ppv_ci_low=0.3, ppv_ci_high=0.5, calib_slope=1.0312,
        calib_intercept=0.0161,
    )  # fmt: skip


def _dl(a: str, b: str, aa: float, ab: float, p: float) -> dict:
    return dict(a=a, b=b, auc_a=aa, auc_b=ab, diff=aa - ab, se_diff=0.01, z=-1.9, p=p,
                ci_low=-0.02, ci_high=0.001)  # fmt: skip


@pytest.fixture
def metrics() -> dict:
    aucs = {"logit": 0.81, "rf": 0.82, "hgb": 0.83}
    return {
        "test": {k: _test(v) for k, v in aucs.items()},
        "cv": {k: dict(cv_auc_mean=v + 0.01, cv_auc_sd=0.01) for k, v in aucs.items()},
        "best_params": {
            "logit": {"model__C": 0.1},
            "rf": {"model__max_depth": None, "model__n_estimators": 300},
            "hgb": {"model__learning_rate": 0.05},
        },
        "delong": [
            _dl("logit", "rf", 0.81, 0.82, 0.051),
            _dl("logit", "hgb", 0.81, 0.83, 0.001),
            _dl("rf", "hgb", 0.82, 0.83, 0.2),
        ],
        "best_model": "hgb",
        "findrisc": {
            k: dict(auc=0.78, ci_low=0.75, ci_high=0.8, n=100, n_pos=10, valid=k != "c")
            for k in "abc"
        },
        "delong_vs_findrisc": [
            dict(
                model="hgb",
                variant="a",
                n=100,
                auc_a=0.83,
                auc_b=0.78,
                diff=0.05,
                se_diff=0.01,
                z=6.0,
                p=1e-9,
                ci_low=0.03,
                ci_high=0.06,
            ),
        ],  # fmt: skip
        "calibration": {
            k: {"bin": [1, 2], "mean_pred": [0.1, 0.5], "frac_pos": [0.1, 0.6], "n": [5, 5]}
            for k in aucs
        },  # fmt: skip
    }


def test_metrics_summary_shape_and_formats(metrics: dict) -> None:
    t = metrics_summary_table(metrics)
    assert list(t.index) == ["Логістична регресія", "Random Forest", "Gradient Boosting"]
    assert t.shape == (3, 11)
    row = t.loc["Логістична регресія"]
    assert row["AUC крос-валідації (середнє ± SD)"] == "0.820 ± 0.010"
    assert row["95% ДІ AUC (тест)"] == "0.790–0.830"
    assert row["Оцінка Браєра"] == 0.12 and row["Чутливість (recall)"] == 0.8
    assert row["Обрані параметри"] == "обернена сила регуляризації C: 0.1"
    rf = t.loc["Random Forest", "Обрані параметри"]
    assert rf == "глибина дерев: без обмеження; n_estimators: 300"
    hgb = t.loc["Gradient Boosting", "Обрані параметри"]
    assert hgb == "швидкість навчання: 0.05"


def test_metrics_ci_table(metrics: dict) -> None:
    t = metrics_ci_table(metrics)
    assert t["Модель"].tolist() == ["Логістична регресія", "Random Forest", "Gradient Boosting"]
    row = t.iloc[1]
    assert row["Чутливість, %"] == 80.0 and row["Чутливість: 95% ДІ, %"] == "75.2–82.4"
    assert row["PPV: 95% ДІ, %"] == "30.0–50.0"
    assert (row["Нахил калібрування"], row["Зсув калібрування (log-odds)"]) == (1.031, 0.016)
    assert pct_ci(0.1234, 0.5) == "12.3–50.0"


def test_delong_table(metrics: dict) -> None:
    t = delong_table(metrics)
    assert len(t) == 3
    # Холм: 0.001·3 = 0.003; 0.051·2 = 0.102; 0.2 → висновок за скоригованим p
    assert t["p (Холм)"].tolist() == ["0.102", "0.003", "0.200"]
    assert t["Висновок"].tolist() == ["незначуща", "значуща", "незначуща"]
    metrics["delong"][0]["p"] = 0.02  # сире p < 0.05, але 0.02·2 = 0.04 → значуща
    metrics["delong"][2]["p"] = 0.03  # 0.03·1, проте монотонність Холма → 0.04
    assert delong_table(metrics)["Висновок"].tolist() == ["значуща"] * 3
    metrics["delong"][0]["p"] = 0.03  # 0.03·2 = 0.06 — сира значущість губиться
    assert delong_table(metrics).loc[0, "Висновок"] == "незначуща"


def test_findrisc_tables(metrics: dict) -> None:
    t = findrisc_auc_table(metrics)
    assert t["Варіант"].tolist() == ["a", "b", "c"]
    assert t["Придатний для порівняння"].tolist() == ["так", "так", "ні"]
    v = findrisc_vs_model_table(metrics)
    assert v.loc[0, "Модель"] == "Gradient Boosting" and v.loc[0, "Висновок"] == "значуща"
    assert v.loc[0, "p (Холм)"] == "< 0.001"


def test_findrisc_vs_model_holm(metrics: dict) -> None:
    row_b = dict(metrics["delong_vs_findrisc"][0], variant="b", p=0.03)
    metrics["delong_vs_findrisc"] = [dict(metrics["delong_vs_findrisc"][0], p=0.04), row_b]
    v = findrisc_vs_model_table(metrics)
    assert v["p"].tolist() == ["0.040", "0.030"]
    assert v["p (Холм)"].tolist() == ["0.060", "0.060"]
    assert v["Висновок"].tolist() == ["незначуща", "незначуща"]
    assert findrisc_holm(metrics) == pytest.approx([0.06, 0.06])


def test_calibration_and_confusion(metrics: dict) -> None:
    fr = calibration_frames(metrics)
    assert set(fr) == {"logit", "rf", "hgb"} and list(fr["rf"].columns)[:3] == [
        "bin", "mean_pred", "frac_pos",
    ]  # fmt: skip
    assert confusion_counts(metrics, "rf") == (10, 4, 2, 6)


def test_conclusion_names_models_and_rule(metrics: dict) -> None:
    text = best_model_conclusion(metrics)
    assert text.count("Gradient Boosting: AUC 0.830;") == 2  # обрана модель — першою
    assert "Різниця +0.020 є статистично" in text and "Різниця +0.010 не є" in text
    assert "з поправкою p = 0.003" in text and "з поправкою p = 0.200" in text
    assert "Логістична регресія" in text and "Random Forest" in text
    assert text.startswith("З поправкою Холма на кількість порівнянь (3)")
    assert "значуща порівняно з: «Логістична регресія»." in text
    assert "крос-валідації" not in text
    metrics["best_model_rule"] = "cv"
    assert "крос-валідації" in best_model_conclusion(metrics)
    assert "Вищий тестовий AUC" not in best_model_conclusion(metrics)  # hgb найвища і на тесті
    metrics["best_model"] = "rf"
    assert "Вищий тестовий AUC в іншої моделі" in best_model_conclusion(metrics)


def test_conclusion_when_nothing_significant_after_holm(metrics: dict) -> None:
    # сирі p 0.024 і 0.048 після Холма стають 0.072 і 0.096 — як на реальних даних
    for r, p in zip(metrics["delong"], (0.717, 0.024, 0.048), strict=True):
        r["p"] = p
    text = best_model_conclusion(metrics)
    assert "статистично не відрізняється за AUC від жодної з інших моделей" in text
    assert "Без поправки вона виглядала б значущою." in text
    assert text.count("не є статистично значущою") == 2
    assert delong_holm(metrics) == pytest.approx([0.717, 0.072, 0.096])


def test_pair_text_missing_pair_raises(metrics: dict) -> None:
    metrics["delong"] = []
    with pytest.raises(DataValidationError, match="DeLong"):
        _pair_text(metrics, "hgb", "rf")


# ---------- підгрупи: діагностований і недіагностований ЦД2 ----------
@pytest.fixture
def preds() -> pd.DataFrame:
    """Синтетичний тест: 40 без ЦД2, 12 діагностованих, 8 недіагностованих."""
    rng = np.random.default_rng(1)
    n0, nd, nu = 40, 12, 8
    t2d = np.r_[np.zeros(n0), np.ones(nd + nu)]
    diq = np.r_[np.full(n0, 2.0), np.ones(nd), np.full(nu, 2.0)]
    signal = t2d + rng.normal(0, 0.8, len(t2d))
    frd = np.round(signal * 4 + 8)
    frd[:3] = np.nan  # анкета заповнена не в усіх
    return pd.DataFrame({
        "SEQN": np.arange(len(t2d)), "t2d": t2d, "diq010": diq,
        "p_logit": signal + rng.normal(0, 0.5, len(t2d)), "p_rf": signal,
        "p_hgb": signal + rng.normal(0, 0.2, len(t2d)), "findrisc_a": frd,
    })  # fmt: skip


def test_subgroup_split(preds: pd.DataFrame) -> None:
    undiag, diag = subgroup(preds, "undiag"), subgroup(preds, "diag")
    assert (len(undiag), int(undiag["t2d"].sum())) == (48, 8)
    assert (len(diag), int(diag["t2d"].sum())) == (52, 12)
    assert (diag.loc[diag["t2d"] == 1, "diq010"] == 1).all()


def test_subgroup_auc_rows_and_holm(preds: pd.DataFrame) -> None:
    raw = subgroup_auc(preds, "hgb")
    assert len(raw) == 8 and set(raw["group"]) == {"undiag", "diag"}
    compared = raw.dropna(subset=["p"])
    assert sorted(compared["score"]) == ["findrisc_a", "findrisc_a", "logit", "logit"]
    assert (compared["p_holm"] >= compared["p"] - 1e-12).all()
    frd = raw[(raw["score"] == "findrisc_a") & (raw["group"] == "diag")].iloc[0]
    assert frd["n"] == frd["n_pair"] == 49  # 3 людини без анкети виключено з пари
    best = raw[(raw["score"] == "hgb") & (raw["group"] == "diag")].iloc[0]
    assert best["n"] == 52 and np.isnan(best["p"])


def test_subgroup_view_numbers_and_text(preds: pd.DataFrame) -> None:
    raw = subgroup_auc(preds, "hgb")
    view = subgroup_view(raw, "hgb")
    assert list(view.columns) == [
        "Підгрупа", "Оцінка", "N", "З ЦД2", "AUC", "95% ДІ",
        "Різниця AUC (основна − ця)", "p (ДеЛонга)", "p (Холм)",
    ]  # fmt: skip
    assert "Gradient Boosting (основна)" in set(view["Оцінка"])
    assert set(view.loc[view["Оцінка"] == "Random Forest", "p (Холм)"]) == {"—"}
    numbers = subgroup_numbers(raw)
    expected = {"sub_undiag_hgb_auc", "p_holm_sub_diag_findrisc_a", "sub_undiag_logit_auc_diff"}
    assert expected <= set(numbers) and "p_sub_undiag_rf" not in numbers
    assert isinstance(numbers["sub_diag_findrisc_a_n_pair"], int)
    lines = subgroup_comparisons(raw, "hgb")
    assert len(lines) == 4 and all("з поправкою Холма" in s for s in lines)
    assert lines[0].startswith("Недіагностований ЦД2 — Gradient Boosting: AUC")


def test_subgroup_conclusion_branches(preds: pd.DataFrame) -> None:
    raw = subgroup_auc(preds, "hgb")
    raw["p_holm"] = np.where(raw["p"].notna(), 0.5, np.nan)
    text = subgroup_conclusion(raw, "hgb")
    assert "60 % уже мають діагноз" in text
    assert "не доведена" in text and "лише 8" in text
    assert "«Логістична регресія»" in text and "«FINDRISC (a)»" in text
    raw["p_holm"] = np.where(raw["p"].notna(), 1e-4, np.nan)
    raw["diff"] = np.where(raw["p"].notna(), 0.05, np.nan)
    assert "зберігається і для недіагностованого" in subgroup_conclusion(raw, "hgb")
    # основна модель — логістична регресія: порівняння з самою собою немає
    assert "«Логістична регресія»" not in subgroup_conclusion(subgroup_auc(preds, "logit"), "logit")
