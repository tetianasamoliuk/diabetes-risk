"""Збирання таблиць, рисунків і чисел для документів та експорт у Excel/ZIP."""

import io
import re
import subprocess
import sys
import zipfile

import numpy as np
import openpyxl
import pandas as pd
import pytest

from risk.config import FEATURES
from risk.data.cohort import FlowStep
from risk.external.ukraine import load_us_reference
from risk.models.pipelines import make_hgb
from risk.models.train import load_models
from risk.report.glossary import GLOSSARY
from risk.report.results import (
    SHARE_NO_T2D,
    SHARE_T2D,
    Counter,
    ResultsBundle,
    _join_uk,
    _nonsignificant_phrase,
    build_results,
    export_excel,
    export_zip,
    section_ukraine,
)

N = 300


@pytest.fixture(scope="module")
def inputs() -> dict:
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 80, N)
    bmi = rng.normal(28, 5, N)
    t2d = (
        rng.uniform(size=N) < 1 / (1 + np.exp(-(0.04 * (age - 50) + 0.1 * (bmi - 28) - 1.5)))
    ).astype(float)
    diq010 = np.where((t2d == 1) & (np.arange(N) % 2 == 0), 1.0, 2.0)  # половина хворих з діагнозом
    cohort = pd.DataFrame(
        {
            "SEQN": np.arange(N),
            "cycle": np.where(np.arange(N) % 2 == 0, "2015-2016", "2017-2020"),
            "cycle_2017": (np.arange(N) % 2).astype(float),
            "age": age,
            "male": rng.integers(0, 2, N).astype(float),
            "bmi": bmi,
            "waist": bmi * 3.2 + rng.normal(0, 5, N),
            "sbp": rng.normal(122, 15, N),
            "dbp": rng.normal(72, 10, N),
            "tc": rng.normal(190, 35, N),
            "hdl": rng.normal(52, 14, N),
            "pa_min_week": rng.uniform(0, 400, N),
            "pa_h_week": rng.uniform(0, 7, N),
            "fam_dm": rng.integers(0, 2, N).astype(float),
            "bp_med": rng.integers(0, 2, N).astype(float),
            "chol_med": rng.integers(0, 2, N).astype(float),
            "high_glu_hist": rng.integers(0, 2, N).astype(float),
            "diq010": diq010,
            "hba1c": rng.normal(5.6, 0.8, N),
            "t2d": t2d,
            "t2d_undiag": np.where(diq010 == 1, np.nan, t2d),
            "wt_mec": rng.uniform(1000, 9000, N),
            "psu": np.arange(N) // 2 % 2 + 1.0,
            "strata": np.repeat(np.arange(1.0, 11.0), N // 10),
            "examined": True,
            "pregnant": False,
        }
    )
    flow = [FlowStep("Усі", N + 50, 0), FlowStep("Когорта", N, 50)]
    variables = pd.DataFrame(
        {
            "variable": ["age"], "label_uk": ["Вік, років"], "nhanes_code": ["RIDAGEYR"],
            "unit": ["років"], "missing_pct": [0.0],
        }
    )  # fmt: skip
    y = cohort["t2d"].to_numpy()
    test = cohort.iloc[: N // 4]
    p = np.clip(0.5 * y[: N // 4] + rng.uniform(0, 0.5, N // 4), 0, 1)
    preds = pd.DataFrame(
        {"SEQN": test["SEQN"], "t2d": test["t2d"], "diq010": test["diq010"]}
    ).assign(
        p_logit=p + rng.normal(0, 0.1, N // 4), p_rf=p, p_hgb=p,
        findrisc_a=np.where(np.arange(N // 4) < 5, np.nan, np.round(p * 20)),
    )  # fmt: skip
    hgb = make_hgb().fit(cohort[FEATURES], cohort["t2d"])
    ukraine = pd.DataFrame(
        {
            "source": ["IDF", "IDF", "IDF", "IDF", "МОЗ"], "year": [2024, 2024, 2024, 2024, 2025],
            "indicator": [
                "Поширеність діабету", "Кількість дорослих 20–79 з діабетом",
                "Кількість дорослих 20–79 з недіагностованим діабетом", "Доросле населення 20–79",
                "Пацієнтів із цукровим діабетом",
            ],
            "value": [10.0, 800.0, 300.0, 10000.0, 460.0], "unit": ["%"] + ["осіб"] * 4,
            "url": ["u"] * 5, "accessed": ["2026-01-01"] * 5,
        }
    )  # fmt: skip
    # метрики реальні (models/metrics.json): навчання трьох моделей надто повільне
    return {"cohort": cohort, "flow": flow, "variables": variables, "preds": preds,
            "models": {"hgb": hgb}, "metrics": load_models()[1], "ukraine": ukraine}  # fmt: skip


@pytest.fixture(scope="module")
def bundle(inputs: dict) -> ResultsBundle:
    i = inputs
    return build_results(
        i["cohort"], i["flow"], i["variables"],
        models=i["models"], metrics=i["metrics"], preds=i["preds"], ukraine=i["ukraine"],
    )  # fmt: skip


def test_bundle_content(bundle: ResultsBundle) -> None:
    assert len(bundle.tables) >= 8
    assert len(bundle.figures) >= 6
    assert all(png.startswith(b"\x89PNG") for png in bundle.figures.values())
    assert all(k.startswith("Таблиця ") for k in bundle.tables)
    assert "Таблиця 1.2. Змінні дослідження" in bundle.tables
    assert "Рисунок 1.1. Блок-схема відбору учасників до аналітичної вибірки" in bundle.figures
    assert bundle.numbers["n_cohort"] == N
    assert bundle.numbers["n_cycle_2015"] + bundle.numbers["n_cycle_2017"] == N


def test_export_excel(bundle: ResultsBundle) -> None:
    wb = openpyxl.load_workbook(io.BytesIO(export_excel(bundle)))
    assert all(len(n) <= 31 for n in wb.sheetnames)
    assert {"Графіки", "Пояснення", "2.1 Описова статистика", "3.2 Гіпотези"} <= set(wb.sheetnames)


def test_export_zip(bundle: ResultsBundle) -> None:
    with zipfile.ZipFile(io.BytesIO(export_zip(bundle))) as z:
        names = z.namelist()
    assert len(names) == len(bundle.figures) == len(set(names))
    # без службової позначки й без «-_» на місці двокрапки
    assert not any("експорті" in n or "-_" in n for n in names)
    assert "Рисунок_4.4._Матриця_помилок_—_Random_Forest,_тестова_вибірка.png" in names


def test_glossary() -> None:
    assert len(GLOSSARY) >= 30
    assert all(term and text.strip() for term, text in GLOSSARY)


def test_glossary_has_no_english_words() -> None:
    """Латиниця в глосарії — лише позначення, скорочення й назви, без англійських слів."""
    allowed = {"AUC", "ROC", "SHAP", "OR", "VIF", "F1", "EPV", "OOF", "HbA1c", "NHANES"}
    allowed |= {"FINDRISC", "Random", "Forest", "Gradient", "Boosting", "Streamlit", "Python"}
    allowed |= {"precision", "p", "r", "t"}
    text = " ".join(f"{term} {text}" for term, text in GLOSSARY)
    assert set(re.findall(r"[A-Za-z][A-Za-z0-9]*", text)) <= allowed


def test_model_and_ukraine_sections(bundle: ResultsBundle) -> None:
    t, f, n = bundle.tables, bundle.figures, bundle.numbers
    assert "Таблиця 4.1. Коефіцієнти логістичної регресії на навчальній вибірці" in t
    assert sum(k.startswith("Таблиця 4.") for k in t) == 5
    assert sum(k.startswith("Рисунок 4.") for k in f) == 7
    assert sum(k.startswith("Таблиця 5.") for k in t) == 6
    assert sum(k.startswith("Рисунок 5.") for k in f) == 2
    assert sum(k.startswith("Таблиця 6.") for k in t) == 2
    assert sum(k.startswith("Рисунок 6.") for k in f) == 1
    assert n["best_model"] in {"logit", "rf", "hgb"}
    assert {"cv_auc_hgb", "test_auc_rf", "n_train", "n_test", "findrisc_auc_a"} <= set(n)
    assert "статистично значущою" in n["best_model_conclusion"] + n["delong_vs_findrisc_a"]
    assert "поправкою Холма" in n["best_model_conclusion"] and "Холма" in n["delong_vs_findrisc_a"]


def test_holm_numbers_and_columns(bundle: ResultsBundle) -> None:
    t, n = bundle.tables, bundle.numbers
    for pair in ("logit_rf", "logit_hgb", "rf_hgb"):
        assert n[f"p_holm_delong_{pair}"] >= n[f"p_delong_{pair}"]
    # реальні metrics.json: 0.717 / 0.024 / 0.048 → за Холмом 0.717 / 0.072 / 0.096
    assert n["p_holm_delong_logit_hgb"] == pytest.approx(3 * n["p_delong_logit_hgb"])
    assert n["n_delong_holm_significant"] == 0
    assert {"p_delong_vs_findrisc_a", "p_holm_delong_vs_findrisc_b"} <= set(n)
    for caption in ("Таблиця 4.3.", "Таблиця 5.4."):
        table = next(v for k, v in t.items() if k.startswith(caption))
        assert "p (Холм)" in table.columns


def test_auc_captions_name_the_sample(bundle: ResultsBundle) -> None:
    t = bundle.tables
    for prefix in ("Таблиця 4.3.", "Таблиця 5.4."):
        assert next(k for k in t if k.startswith(prefix)).endswith(
            "(тест ДеЛонга, тестова вибірка)"
        )
    wb = openpyxl.load_workbook(io.BytesIO(export_excel(bundle)))
    assert wb["4.3 Порівняння AUC моделей"]["A1"].value.endswith("(тест ДеЛонга, тестова вибірка)")
    assert wb["5.4 Модель і FINDRISC"]["A1"].value.endswith("тестова вибірка)")


def test_subgroup_table_and_numbers(bundle: ResultsBundle) -> None:
    t, n = bundle.tables, bundle.numbers
    caption = next(k for k in t if k.startswith("Таблиця 5.6."))
    assert "недіагностованого і діагностованого" in caption
    assert len(t[caption]) == 8 and "p (Холм)" in t[caption].columns
    assert {"sub_undiag_hgb_auc", "sub_diag_findrisc_a_auc_lo", "p_holm_sub_undiag_logit"} <= set(n)
    assert "діагноз" in n["subgroup_conclusion"]
    assert 0 < n["sub_pct_diagnosed_test"] < 100


def test_model_and_ukraine_excel_sheets(bundle: ResultsBundle) -> None:
    wb = openpyxl.load_workbook(io.BytesIO(export_excel(bundle)))
    assert {"4.2 Метрики моделей", "4.4 Пороги Юдена", "6.1 США і Україна"} <= set(wb.sheetnames)
    assert all(len(n) <= 31 for n in wb.sheetnames)
    assert len(wb.sheetnames) == len(set(wb.sheetnames))


def test_numbering_matches_pages(bundle: ResultsBundle) -> None:
    t, f = bundle.tables, bundle.figures
    assert list(t)[:4] == [
        "Таблиця 1.1. Кроки відбору учасників",
        "Таблиця 1.2. Змінні дослідження",
        "Таблиця 1.3. Кількість учасників і частка ЦД2 за циклами",
        "Таблиця 1.4. Попередня обробка даних",
    ]
    assert "Таблиця 2.2. Матриця кореляцій Пірсона" in t
    assert "Таблиця 2.3. Матриця кореляцій Спірмена" in t
    assert "Таблиця 2.4. Поширеність ЦД2: загалом і за циклами (незважена та зважена)" in t
    assert not any(k.endswith("(лише в експорті)") for k in t)
    assert "Рисунок 2.1. Розподіл ознаки за групами: Вік, років" in f
    assert "Рисунок 2.6. Розподіл ознаки за групами: Індекс маси тіла, кг/м² (лише в експорті)" in f
    assert "Рисунок 2.5. Поширеність ЦД2: незважена та зважена оцінки" in f
    assert "Таблиця 3.1. Нульова та альтернативна гіпотези для кожного фактора" in t
    assert "Таблиця 3.4. Відношення шансів за клінічними порогами" in t
    assert "Таблиця 3.6. Ранжування факторів за багатофакторною моделлю" in t
    assert any(k.startswith("Таблиця 3.7. Відношення шансів загального холестерину") for k in t)
    assert "Рисунок 3.2. Форест-графік (forest plot) багатофакторних відношень шансів" in f
    assert "Таблиця 4.4. Пороги Юдена та відповідні чутливість і специфічність" in t
    assert "Таблиця 5.5. AUC FINDRISC на всій аналітичній вибірці (описово)" in t
    assert "Таблиця 6.2. Інші показники" in t
    assert "Рисунок 4.7. SHAP: середня важливість ознак у прогнозі Gradient Boosting" in f


def test_effect_sizes_in_tests_table(bundle: ResultsBundle) -> None:
    t = bundle.tables["Таблиця 3.2. Порівняння груп без ЦД2 і з ЦД2: тести та висновки"]
    auc, r = t["AUC (ймовірність вищого значення у хворого)"], t["r (рангово-бісеріальна)"]
    ok = auc.notna()
    assert ok.any() and (r[ok] - (2 * auc[ok] - 1)).abs().max() < 2e-3
    # частки для бінарних ознак; для неперервних — «—»
    shares = t.set_index("Фактор")[[SHARE_NO_T2D, SHARE_T2D]]
    assert (shares.loc["Вік, років"] == "—").all()
    assert all(0 <= float(x) <= 100 for x in shares.loc["Стать (чоловіча)"])


def test_template_numbers(bundle: ResultsBundle) -> None:
    n = bundle.numbers
    for f in ("age", "waist", "hdl", "fam_dm", "bp_med", "male"):
        assert {f"or_uni_{f}", f"or_uni_{f}_lo", f"or_uni_{f}_hi"} <= set(n)
    assert {"or_clin_age", "or_clin_bmi_lo", "or_clin_bmi_hi"} <= set(n)
    assert {"or_multi_bmi", "p_multi_bmi", "vif_bmi", "vif_waist", "v_statsmodels"} <= set(n)
    assert {"weighted_prev_2015_lo", "weighted_prev_2017_hi", "pct_diagnosed"} <= set(n)
    assert isinstance(n["tests_nonsignificant"], list)
    assert len(n["shap_top3"]) == 3
    new = {
        "top3_factors", "tests_nonsignificant_phrase", "or_tc_all", "or_tc_no_cholmed",
        "or_tc_no_cholmed_p", "n_no_cholmed", "cholmed_pct_t2d", "cholmed_pct_no_t2d",
        "prev_20_79", "deff_prev", "n_eff_prev", "ppv_at_us_prev", "ppv_at_ua_prev",
        "n_logit_train", "frd_cohort_auc_a", "frd_cohort_auc_b", "frd_cohort_n_b",
        "frd_cohort_pos_b", "frd_a_excluded_n", "frd_a_excluded_pos_pct",
        "nhanes_prev_diag", "nhanes_prev_diag_lo", "nhanes_prev_undiag_hi", "ua_prev_crude",
        "ua_prev_diag", "ua_prev_undiag", "ua_prev_moz", "ua_idf_diag_n", "ua_moz_n",
        "ua_moz_vs_idf_pct", "ua_moz_idf_text", "ppv_at_ua_prev_basis",
    }  # fmt: skip
    assert new <= set(n)
    assert n["ua_prev_crude"] == pytest.approx(8.0) and n["ua_moz_vs_idf_pct"] == pytest.approx(92)
    assert (n["ua_idf_diag_n"], n["ua_moz_n"]) == (500, 460)
    assert n["frd_a_excluded_n"] == 5
    assert n["top3_factors"].count(",") == 1 and " і " in n["top3_factors"]
    assert n["or_tc_all"].count("(") == 1 and "–" in n["or_tc_all"]


def test_excel_sheet_names(bundle: ResultsBundle) -> None:
    wb = openpyxl.load_workbook(io.BytesIO(export_excel(bundle)))
    expected = {
        "1.1 Кроки відбору", "1.3 Учасники за циклами", "1.4 Попередня обробка",
        "3.6 Ранжування факторів", "4.5 Довірчі інтервали метрик", "6.2 Інші показники",
    }  # fmt: skip
    assert expected <= set(wb.sheetnames)
    corr = wb["2.2 Кореляції Пірсона"]
    assert corr.cell(row=3, column=2).number_format == "0.000"


def test_section_ukraine_empty() -> None:
    empty = pd.DataFrame()
    assert section_ukraine(empty, empty, empty, {}, Counter(6)) == ({}, {}, {})


def test_section_ukraine_without_main_prevalence(inputs: dict) -> None:
    only_moz = inputs["ukraine"].iloc[4:]
    us = load_us_reference()
    tables, figs, numbers = section_ukraine(
        inputs["cohort"], only_moz, us, inputs["metrics"], Counter(6)
    )
    first = next(iter(tables.values()))
    assert set(first["Джерело"]) == {"США (NHANES, ця робота)", "США (NHANES)", "США (NCHS)"}
    assert list(first["Показник"]).count("Діагностований діабет") == 2
    assert numbers == {} and len(figs) == 1


def test_comparison_table_has_dash_instead_of_none(bundle: ResultsBundle) -> None:
    table = bundle.tables[
        "Таблиця 6.1. Поширеність діабету серед дорослих: США і Україна (різні методи)"
    ]
    ci = table[["95% ДІ від, %", "95% ДІ до, %"]].astype(str).to_numpy().ravel()
    assert "—" in ci and not {"None", "nan", "NaN"} & set(ci)
    assert table["Метод"].str.contains("розрахунок із чисел джерела").sum() == 4
    assert "Україна (МОЗ 2025)" in set(table["Джерело"])


def test_results_module_does_not_import_shap() -> None:
    # сторінки імпортують results через _common; shap потрібен лише для розділу 4
    code = "import sys, risk.report.results; sys.exit('shap' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_phrases() -> None:
    assert _join_uk(["вік"]) == "вік"
    assert _join_uk(["вік", "ІМТ", "талія"]) == "вік, ІМТ і талія"
    assert _nonsignificant_phrase([]) == "таких факторів немає"
    assert _nonsignificant_phrase(["Діастолічний АТ"]) == "за одним фактором — діастолічний АТ"
    assert _nonsignificant_phrase(["ІМТ", "Вік"]) == "за 2 факторами — ІМТ і вік"
