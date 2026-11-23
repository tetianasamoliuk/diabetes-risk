"""Єдине місце збирання таблиць, рисунків і ключових чисел для Excel та документа «Результати».

Розділи відповідають сторінкам застосунку: 1 — дані і вибірка, 2 — статистика, 3 — фактори
ризику, 4 — моделі, 5 — FINDRISC, 6 — порівняння з Україною. Кожна `section_*` повертає
(tables, figures, numbers); нумерація таблиць і рисунків збігається з порядком показу на сторінках.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels
from matplotlib.figure import Figure

from risk import interpret
from risk.config import (
    ADULT_AGE,
    ALPHA,
    BINARY_FACTORS,
    CONTINUOUS_FACTORS,
    CYCLES,
    FEATURE_LABELS_UK,
    FEATURES,
    HBA1C_THRESHOLD,
    MODEL_NAMES_UK,
)
from risk.data.cohort import FlowStep
from risk.data.preprocessing import preprocessing_table
from risk.data.weights import design_effect, prevalence
from risk.external.ukraine import (
    CALC_NOTE,
    COMPARISON_COLS_UK,
    COMPARISON_COLUMNS,
    MEASURE_DIAG,
    MEASURE_TOTAL,
    MEASURE_UNDIAG,
    OTHER_COLS_UK,
    load_us_reference,
    moz_idf_text,
    moz_vs_idf,
    other_indicators,
    prevalence_comparison,
    ukraine_counts,
    ukraine_shares,
)
from risk.findrisc.score import (
    MAX_SCORE_A,
    ORIGINAL_VS_MODIFIED,
    FindriscVariant,
    categories,
    category_table,
    variants,
)
from risk.models.delong import auc_ci
from risk.models.evaluate import ppv_at_prevalence, roc_points
from risk.models.summary import (
    SUBGROUP_CAPTION,
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
    pct_diagnosed,
    subgroup_auc,
    subgroup_conclusion,
    subgroup_numbers,
    subgroup_view,
)
from risk.report.excel import ExcelSheet, build_workbook, figures_zip
from risk.report.glossary import glossary_frame
from risk.stats import odds
from risk.stats.correlation import Method, corr_matrix
from risk.stats.descriptive import describe_by_group
from risk.stats.hypothesis import run_all_tests
from risk.viz import figures as viz
from risk.viz.style import GROUP_LABELS, close, fig_to_png

if TYPE_CHECKING:  # shap важкий: під час виконання імпортується лише в section_models
    import shap

type Number = float | int | str | list[str]
type Part = tuple[dict[str, pd.DataFrame], dict[str, bytes], dict[str, Number]]

FACTORS = [f for f in FEATURES if f != "cycle_2017"]  # індикатор циклу — не фактор ризику
BOX_FACTORS = ["bmi", "waist", "sbp", "hdl", "age", "tc"]
ONLY_EXPORT = " (лише в експорті)"  # елемент є в Excel, але не показаний на сторінці
HIST_DEFAULT = CONTINUOUS_FACTORS[0]  # ознака гістограми за замовчуванням на сторінці 2
HIST_EXTRA = "bmi"  # друга гістограма — лише в експорті
CORR_METHODS: tuple[tuple[Method, str], ...] = (("pearson", "Пірсона"), ("spearman", "Спірмена"))
AGE_TOPCODE = 80  # NHANES записує вік 80 і більше як 80, тому «20–79» — це age < 80
NHANES_METHOD = "HbA1c + самозвіт"
COMPARISON_CAPTION = "Поширеність діабету серед дорослих: США і Україна (різні методи)"
# Початок підпису порогу в clinical_or_table -> колонка-джерело (для ключів or_clin_<col>)
CLIN_COLS = {"Вік": "age", "ІМТ": "bmi", "САТ": "sbp", "Талія": "waist", "Активність": "pa_h_week"}
OR_COLS_UK = {"odds_ratio": "OR", "ci_low": "95% ДІ від", "ci_high": "95% ДІ до", "p": "p"}
DESCRIBE_COLS_UK = {
    "variable": "Змінна", "group": "Група", "n": "N", "mean": "Середнє", "sd": "SD",
    "median": "Медіана", "q1": "Q1", "q3": "Q3", "iqr": "IQR", "min": "Мін", "max": "Макс",
}  # fmt: skip
PREV_COLS_UK = {
    "group": "Група", "n": "N", "unweighted_pct": "Незважена частка, %",
    "weighted_pct": "Зважена частка, %", "ci_low": "95% ДІ від, %", "ci_high": "95% ДІ до, %",
}  # fmt: skip
ORIGINAL_COLS_UK = {
    "item": "Пункт", "original": "Оригінал", "modified": "Модифікований",
    "change": "Що змінено і чому",
}  # fmt: skip
CATEGORY_COLS_UK = {
    "category": "Категорія ризику", "n": "N", "n_t2d": "З ЦД2", "pct_t2d": "% з ЦД2",
}  # fmt: skip
CHOL_COLS_UK = {
    "group": "Група", "n": "N", "n_pos": "З ЦД2", **OR_COLS_UK,
}  # fmt: skip
COHORT_FINDRISC = {
    "a": "(a) 6 пунктів без «глюкози в анамнезі»; уся аналітична вибірка з повною анкетою",
    "b": "(b) скринінг: з пунктом про глюкозу, лише недіагностовані",
}
# Назви листів Excel (≤ 31 символу) за номером таблиці; для невідомих — номер + підпис
SHEET_NAMES = {
    "1.1": "1.1 Кроки відбору",
    "1.2": "1.2 Змінні",
    "1.3": "1.3 Учасники за циклами",
    "1.4": "1.4 Попередня обробка",
    "2.1": "2.1 Описова статистика",
    "2.2": "2.2 Кореляції Пірсона",
    "2.3": "2.3 Кореляції Спірмена",
    "2.4": "2.4 Поширеність ЦД2",
    "3.1": "3.1 Гіпотези H0 і H1",
    "3.2": "3.2 Гіпотези",
    "3.3": "3.3 Відношення шансів",
    "3.4": "3.4 Клінічні пороги OR",
    "3.5": "3.5 Багатофакторна модель",
    "3.6": "3.6 Ранжування факторів",
    "3.7": "3.7 Холестерин і ліки",
    "4.1": "4.1 Коефіцієнти лог. регресії",
    "4.2": "4.2 Метрики моделей",
    "4.3": "4.3 Порівняння AUC моделей",
    "4.4": "4.4 Пороги Юдена",
    "4.5": "4.5 Довірчі інтервали метрик",
    "5.1": "5.1 FINDRISC оригінал-модиф.",
    "5.2": "5.2 Категорії FINDRISC",
    "5.3": "5.3 AUC варіантів FINDRISC",
    "5.4": "5.4 Модель і FINDRISC",
    "5.5": "5.5 AUC FINDRISC на вибірці",
    "5.6": "5.6 Підгрупи за діагнозом",
    "6.1": "6.1 США і Україна",
    "6.2": "6.2 Інші показники",
}
CORR_FORMAT = "0.000"  # коефіцієнти кореляції в Excel — три знаки
# Частки бінарних ознак у групах (колонки таблиці 3.2)
SHARE_NO_T2D, SHARE_T2D = "Частка з ознакою: без ЦД2, %", "Частка з ознакою: з ЦД2, %"
SHAP_MODEL = "hgb"  # SHAP рахується для Gradient Boosting (дерева без імпутера)


@dataclass(frozen=True, slots=True)
class ResultsBundle:
    """Усе, що йде в Excel і документ: таблиці, рисунки (PNG), ключові числа, глосарій."""

    tables: dict[str, pd.DataFrame]
    figures: dict[str, bytes]
    numbers: dict[str, Number]
    glossary: pd.DataFrame


class Counter:
    """Лічильник таблиць і рисунків сторінки: «Таблиця 1.1.», «Рисунок 2.3.»."""

    def __init__(self, page: int) -> None:
        """Нумерація таблиць і рисунків сторінки page починається з 1."""
        self.page, self._t, self._f = page, 0, 0

    def table(self, caption: str, *, export_only: bool = False) -> str:
        """Наступний підпис таблиці «Таблиця N.M. …» (export_only — позначка «лише в експорті»)."""
        self._t += 1
        return f"Таблиця {self.page}.{self._t}. {caption}{ONLY_EXPORT if export_only else ''}"

    def figure(self, caption: str, *, export_only: bool = False) -> str:
        """Наступний підпис рисунка «Рисунок N.M. …» (export_only — позначка «лише в експорті»)."""
        self._f += 1
        return f"Рисунок {self.page}.{self._f}. {caption}{ONLY_EXPORT if export_only else ''}"


def _png(fig: Figure) -> bytes:
    png = fig_to_png(fig, 300)
    close(fig)
    return png


def _n_in_cycle(df: pd.DataFrame, cycle: str) -> int:
    return int((df["cycle"] == cycle).sum())


def _short(feature: str) -> str:
    """Українська назва ознаки без одиниць: «Вік, років» -> «Вік»."""
    return FEATURE_LABELS_UK[feature].split(",")[0]


def _in_sentence(name: str) -> str:
    """Назва всередині речення: «Вік» → «вік», абревіатури («ІМТ», «САТ») без змін."""
    return name[:1].lower() + name[1:] if name[1:2].islower() else name


def _join_uk(items: list[str]) -> str:
    """Перелік через кому з «і» перед останнім: «a, b і c»."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} і {items[-1]}"


def _or_text(r: pd.Series) -> str:
    """«0.94 (0.92–0.96)»."""
    return f"{r['odds_ratio']:.2f} ({r['ci_low']:.2f}–{r['ci_high']:.2f})"


# ---------- 1. Дані і вибірка (сторінка 1) ----------


def prevalence_all(cohort: pd.DataFrame) -> pd.DataFrame:
    """Поширеність: рядок «Разом» + по циклах (англійські колонки, як у prevalence)."""
    return pd.concat([prevalence(cohort), prevalence(cohort, by="cycle")], ignore_index=True)


def prevalence_view(prev: pd.DataFrame) -> pd.DataFrame:
    """Таблиця поширеності з українськими заголовками, до 2 знаків."""
    return prev.rename(columns=PREV_COLS_UK).round(2)


def flow_table(flow: list[FlowStep]) -> pd.DataFrame:
    """Кроки відбору учасників: N на початку кроку і скільки виключено."""
    return pd.DataFrame([{"Крок": s.label, "N": s.n, "Виключено": s.excluded} for s in flow])


def _variables_table(variables: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "label_uk": "Змінна", "nhanes_code": "Код NHANES", "unit": "Одиниці",
        "missing_pct": "Пропуски, %",
    }  # fmt: skip
    return variables[list(cols)].rename(columns=cols)


def _sample_numbers(cohort: pd.DataFrame, flow: list[FlowStep], prev: pd.DataFrame) -> dict:
    total = prev.iloc[0]
    positives = cohort[cohort["t2d"] == 1]
    numbers: dict[str, Number] = {
        "n_cohort": len(cohort),
        "n_pos": int(cohort["t2d"].sum()),
        "pct_pos": float(cohort["t2d"].mean() * 100),
        "pct_diagnosed": float((positives["diq010"] == 1).mean() * 100),
        "weighted_prev": float(total["weighted_pct"]),
        "weighted_ci_low": float(total["ci_low"]),
        "weighted_ci_high": float(total["ci_high"]),
        "n_flow_start": flow[0].n,
        **{f"n_cycle_{c['year']}": _n_in_cycle(cohort, k) for k, c in CYCLES.items()},
    }
    by_group = prev.set_index("group")
    for k, c in CYCLES.items():
        row, key = by_group.loc[k], f"weighted_prev_{c['year']}"
        numbers |= {key: float(row["weighted_pct"]), f"{key}_lo": float(row["ci_low"]),
                    f"{key}_hi": float(row["ci_high"])}  # fmt: skip
    return numbers


def section_sample(
    cohort: pd.DataFrame, flow: list[FlowStep], variables: pd.DataFrame, num: Counter
) -> Part:
    """Сторінка 1: кроки відбору, змінні, цикли, попередня обробка; блок-схема і частка ЦД2."""
    prev = prevalence_all(cohort)
    by_cycle = prev[prev["group"] != "Разом"].reset_index(drop=True)
    tables = {
        num.table("Кроки відбору учасників"): flow_table(flow),
        num.table("Змінні дослідження"): _variables_table(variables),
        num.table("Кількість учасників і частка ЦД2 за циклами"): prevalence_view(by_cycle),
        num.table("Попередня обробка даних"): preprocessing_table(),
    }
    figs = {
        num.figure("Блок-схема відбору учасників до аналітичної вибірки"): _png(
            viz.flowchart(flow)
        ),
        num.figure("Частка ЦД2 за циклами: незважена та зважена оцінки"): _png(
            viz.prevalence_bars(by_cycle)
        ),
    }
    return tables, figs, _sample_numbers(cohort, flow, prev)


# ---------- 2. Статистика (сторінка 2) ----------


def _describe_table(cohort: pd.DataFrame) -> pd.DataFrame:
    t = describe_by_group(cohort, CONTINUOUS_FACTORS).reset_index()
    t["variable"] = t["variable"].map(FEATURE_LABELS_UK)
    t["group"] = t["group"].map(lambda g: GROUP_LABELS[int(g)])
    return t.rename(columns=DESCRIBE_COLS_UK)


def _hist_fig(
    cohort: pd.DataFrame, col: str, num: Counter, *, export_only: bool = False
) -> dict[str, bytes]:
    label = FEATURE_LABELS_UK[col]
    caption = num.figure(f"Розподіл ознаки за групами: {label}", export_only=export_only)
    return {caption: _png(viz.hist_by_group(cohort, col, label))}


def section_descriptive(cohort: pd.DataFrame, num: Counter) -> Part:
    """Описова статистика, гістограма (ознака за замовчуванням сторінки) і boxplot."""
    tables = {num.table("Описова статистика неперервних ознак за групами"): _describe_table(cohort)}
    figs = _hist_fig(cohort, HIST_DEFAULT, num)
    labels = [FEATURE_LABELS_UK[c] for c in BOX_FACTORS]
    fig = viz.box_by_group(cohort, BOX_FACTORS, labels)
    figs[num.figure("Діаграми розмаху (boxplot) ключових ознак за групами")] = _png(fig)
    return tables, figs, {}


def section_correlations(cohort: pd.DataFrame, num: Counter) -> Part:
    """Матриці кореляцій Пірсона і Спірмена: таблиці й heatmap-и."""
    tables, figs = {}, {}
    for method, name in CORR_METHODS:
        corr = corr_matrix(cohort, CONTINUOUS_FACTORS, method)
        shown = corr.rename(index=FEATURE_LABELS_UK, columns=FEATURE_LABELS_UK)
        tables[num.table(f"Матриця кореляцій {name}")] = shown.reset_index(names="Змінна")
        figs[num.figure(f"Матриця кореляцій {name}")] = _png(
            viz.corr_heatmap(corr, f"Кореляційна матриця ({name})")
        )
    return tables, figs, {}


def with_status(cohort: pd.DataFrame) -> pd.DataFrame:
    """Колонки dm_diag (ЦД2 з уже встановленим діагнозом) і dm_undiag (ЦД2 без діагнозу)."""
    sick, diagnosed = cohort["t2d"] == 1, cohort["diq010"] == 1
    return cohort.assign(
        dm_diag=(sick & diagnosed).astype(float), dm_undiag=(sick & ~diagnosed).astype(float)
    )


def _weighted_triplet(key: str, data: pd.DataFrame, outcome: str) -> dict[str, Number]:
    """{key: зважена %, key_lo, key_hi} для outcome."""
    p = prevalence(data, outcome).iloc[0]
    return {key: float(p["weighted_pct"]), f"{key}_lo": float(p["ci_low"]),
            f"{key}_hi": float(p["ci_high"])}  # fmt: skip


def _prevalence_numbers(cohort: pd.DataFrame) -> dict[str, Number]:
    """Зважена поширеність: 20–79 років (як у IDF), діагностований і недіагностований ЦД2."""
    status = with_status(cohort)
    deff, n_eff = design_effect(cohort)
    return {
        **_weighted_triplet("prev_20_79", cohort[cohort["age"] < AGE_TOPCODE], "t2d"),
        **_weighted_triplet("nhanes_prev_diag", status, "dm_diag"),
        **_weighted_triplet("nhanes_prev_undiag", status, "dm_undiag"),
        "deff_prev": deff,
        "n_eff_prev": round(n_eff),
    }


def section_prevalence(cohort: pd.DataFrame, num: Counter) -> Part:
    """Поширеність ЦД2: загалом і за циклами (таблиця і стовпчики сторінки 2)."""
    prev = prevalence_all(cohort)
    caption = "Поширеність ЦД2: загалом і за циклами (незважена та зважена)"
    tables = {num.table(caption): prevalence_view(prev)}
    figs = {
        num.figure("Поширеність ЦД2: незважена та зважена оцінки"): _png(viz.prevalence_bars(prev))
    }
    return tables, figs, _prevalence_numbers(cohort)


def section_stats_extra(cohort: pd.DataFrame, num: Counter) -> Part:
    """Лише в експорті: гістограма другої ознаки."""
    return {}, _hist_fig(cohort, HIST_EXTRA, num, export_only=True), {}


# ---------- 3. Фактори ризику: гіпотези і відношення шансів (сторінка 3) ----------


def share_cells(t: pd.DataFrame) -> dict[str, pd.Series]:
    """Колонки часток бінарних ознак по групах (з run_all_tests); «—» для неперервних."""
    return {
        name: t[col].map(lambda x: interpret.fmt_cell(x, 1))
        for name, col in ((SHARE_NO_T2D, "pct0"), (SHARE_T2D, "pct1"))
    }


def _tests_view(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Фактор": t["label"],
            "Тест": t["test"].map(interpret.test_name_uk),
            "Статистика": t["statistic"].round(2),
            "p": t["p"].map(interpret.fmt_p),
            "p (Холм)": t["p_holm"].map(interpret.fmt_p),
            "AUC (ймовірність вищого значення у хворого)": t["auc"].round(3),
            "r (рангово-бісеріальна)": t["r_rb"].round(3),
            **share_cells(t),
            "Висновок": t["reject_h0"].map({True: "відхиляємо H0", False: "не відхиляємо H0"}),
        }
    )


def _nonsignificant_phrase(labels: list[str]) -> str:
    """«за одним фактором — діастолічний АТ» / «за 2 факторами — …» / «таких факторів немає»."""
    names = _join_uk([_in_sentence(x) for x in labels]) if labels else ""
    match len(labels):
        case 0:
            return "таких факторів немає"
        case 1:
            return f"за одним фактором — {names}"
        case n:
            return f"за {n} факторами — {names}"


def section_hypotheses(cohort: pd.DataFrame, num: Counter) -> Part:
    """Гіпотези H0/H1 і тести порівняння груп з поправкою Холма та розміром ефекту."""
    cat = [c for c in BINARY_FACTORS if c != "cycle_2017"]
    t = run_all_tests(cohort, CONTINUOUS_FACTORS, cat)
    texts = t[["label", "h0_text", "h1_text"]].rename(
        columns={"label": "Фактор", "h0_text": "H0", "h1_text": "H1"}
    )
    tables = {
        num.table("Нульова та альтернативна гіпотези для кожного фактора"): texts,
        num.table("Порівняння груп без ЦД2 і з ЦД2: тести та висновки"): _tests_view(t),
    }
    nonsig = [x.split(",")[0] for x in t.loc[~t["reject_h0"], "label"]]
    numbers: dict[str, Number] = {
        "n_tests": len(t),
        "n_tests_significant": int(t["reject_h0"].sum()),
        "tests_nonsignificant": nonsig,
        "tests_nonsignificant_phrase": _nonsignificant_phrase(nonsig),
        "p_holm_dbp": float(t.loc[t["variable"] == "dbp", "p_holm"].iloc[0]),
    }
    return tables, {}, numbers


def _fmt_or(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Українські назви факторів замість індексу."""
    out = t[cols].copy()
    out.insert(0, "Фактор", [FEATURE_LABELS_UK.get(i, i) for i in out.index])
    return out.reset_index(drop=True)


def _uni_view(uni: pd.DataFrame) -> pd.DataFrame:
    """Однофакторні OR; підпис фактора з кроком («Вік, на 10 років»), як на сторінці 3."""
    out = _fmt_or(uni, list(OR_COLS_UK)).rename(columns=OR_COLS_UK)
    out["Фактор"] = uni["label"].to_numpy()
    return out


def _clinical_view(clin: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "label": "Поріг", "odds_ratio": "OR", "ci_low": "95% ДІ від", "ci_high": "95% ДІ до",
        "p": "p", "n_exposed": "N з експозицією", "n": "N",
    }  # fmt: skip
    return clin[list(cols)].rename(columns=cols)


def _or_triplet(prefix: str, r: pd.Series) -> dict[str, Number]:
    return {
        prefix: float(r["odds_ratio"]),
        f"{prefix}_lo": float(r["ci_low"]),
        f"{prefix}_hi": float(r["ci_high"]),
    }


def _top3_factors(ranked: pd.DataFrame) -> str:
    """Три провідні фактори за стандартизованим ефектом: «вік, … і окружність талії»."""
    return _join_uk([_in_sentence(_short(str(f))) for f in ranked.index[:3]])


def _odds_numbers(
    uni: pd.DataFrame, clin: pd.DataFrame, multi: pd.DataFrame, ranked: pd.DataFrame
) -> dict[str, Number]:
    numbers: dict[str, Number] = {
        "n_multivariable": int(multi.attrs["n"]),
        "top_factor": FEATURE_LABELS_UK.get(str(ranked.index[0]), str(ranked.index[0])),
        "top3_factors": _top3_factors(ranked),
        "or_multi_bmi": float(multi.loc["bmi", "odds_ratio"]),
        "p_multi_bmi": float(multi.loc["bmi", "p"]),
        "vif_bmi": float(multi.loc["bmi", "vif"]),
        "vif_waist": float(multi.loc["waist", "vif"]),
        "p_uni_dbp": float(uni.loc["dbp", "p"]),
        "v_statsmodels": statsmodels.__version__,
    }
    for f in (f for f in CONTINUOUS_FACTORS + BINARY_FACTORS if f in uni.index):
        numbers |= _or_triplet(f"or_uni_{f}", uni.loc[f])
    for _, row in clin.iterrows():
        numbers |= _or_triplet(f"or_clin_{CLIN_COLS[row['label'].split()[0]]}", row)
    return numbers


def _cholesterol_numbers(chol: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Number]:
    """OR холестерину загалом і без ліків від холестерину; частки тих, хто їх приймає."""
    everyone, no_med = chol.iloc[0], chol.iloc[1]
    pct_t2d, pct_no_t2d = odds.share_by_outcome(cohort, "chol_med")
    return {
        "or_tc_all": _or_text(everyone),
        "or_tc_no_cholmed": _or_text(no_med),
        "or_tc_no_cholmed_p": interpret.fmt_p(float(no_med["p"])),
        "n_no_cholmed": int(no_med["n"]),
        "cholmed_pct_t2d": pct_t2d,
        "cholmed_pct_no_t2d": pct_no_t2d,
    }


def _odds_tables(
    uni: pd.DataFrame, clin: pd.DataFrame, multi: pd.DataFrame, num: Counter
) -> dict[str, pd.DataFrame]:
    ranked = odds.rank_factors(multi)
    cols = ["rank", "std_effect", "odds_ratio", "ci_low", "ci_high", "p"]
    return {
        num.table("Однофакторні відношення шансів (кожен фактор окремо)"): _uni_view(uni),
        num.table("Відношення шансів за клінічними порогами"): _clinical_view(clin),
        num.table("Багатофакторна логістична регресія (усі фактори разом)"): _fmt_or(
            multi.drop(index="const"), ["coef", "odds_ratio", "ci_low", "ci_high", "p", "vif"]
        ).rename(columns={"coef": "Коефіцієнт", "vif": "VIF", **OR_COLS_UK}),
        num.table("Ранжування факторів за багатофакторною моделлю"): _fmt_or(ranked, cols).rename(
            columns={"rank": "Місце", "std_effect": "Стандартизований ефект", **OR_COLS_UK}
        ),
    }


def section_odds(cohort: pd.DataFrame, num: Counter) -> Part:
    """Однофакторні та багатофакторні OR, клінічні пороги, ранжування, холестерин і ліки."""
    uni = pd.DataFrame([asdict(odds.univariate_or(cohort, f)) for f in FACTORS])
    uni = uni.set_index("variable")
    clin = odds.clinical_or_table(cohort)
    multi = odds.multivariable_logit(cohort, FEATURES)
    chol = odds.cholesterol_by_medication(cohort)
    tables = _odds_tables(uni, clin, multi, num)
    caption = "Відношення шансів загального холестерину (на 10 мг/дл) і прийом ліків від нього"
    tables[num.table(caption)] = chol.rename(columns=CHOL_COLS_UK)
    per_step = odds.per_clinical_step(multi.drop(index="cycle_2017", errors="ignore"))
    figs = {
        num.figure("Форест-графік (forest plot) однофакторних відношень шансів"): _png(
            viz.forest_plot(uni, "Однофакторні OR з 95% ДІ")
        ),
        num.figure("Форест-графік (forest plot) багатофакторних відношень шансів"): _png(
            viz.forest_plot(per_step, "Багатофакторні OR на клінічний крок, 95% ДІ")
        ),
    }
    numbers = _odds_numbers(uni, clin, multi, odds.rank_factors(multi))
    return tables, figs, numbers | _cholesterol_numbers(chol, cohort)


# ---------- 4. Моделі (сторінка 4) ----------


def coef_view(coefs: pd.DataFrame) -> pd.DataFrame:
    """Таблиця коефіцієнтів логістичної регресії в тому ж вигляді, що на сторінці «Моделі»."""
    view = coefs.drop(index="const")[["coef", "odds_ratio", "ci_low", "ci_high", "p", "vif"]]
    view = view.round(3).assign(p=view["p"].map(interpret.fmt_p))
    view.insert(0, "Ознака", [FEATURE_LABELS_UK.get(i, i) for i in view.index])
    view.columns = ["Ознака", "Коефіцієнт", "OR", "95% ДІ від", "95% ДІ до", "p", "VIF"]
    return view.reset_index(drop=True)


def thresholds_table(metrics: dict) -> pd.DataFrame:
    """Поріг Юдена кожної моделі та чутливість і специфічність на тесті при ньому."""
    rows = {
        MODEL_NAMES_UK[k]: {
            "Поріг (Юден)": round(t["threshold"], 3),
            "Чутливість": round(t["recall"], 3),
            "Специфічність": round(t["specificity"], 3),
        }
        for k, t in metrics["test"].items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Модель").reset_index()


def _shap_top3(expl: shap.Explanation) -> list[str]:
    """Три ознаки з найбільшим середнім |SHAP| (як на стовпчиковому рисунку)."""
    importance = np.abs(np.asarray(expl.values)).mean(axis=0)
    return [_short(expl.feature_names[i]) for i in np.argsort(importance)[::-1][:3]]


SHAP_BAR_CAPTION = f"SHAP: середня важливість ознак у прогнозі {MODEL_NAMES_UK[SHAP_MODEL]}"


def _shap_figures(expl: shap.Explanation, num: Counter) -> dict[str, bytes]:
    return {
        num.figure("SHAP-графік (beeswarm): внесок ознак у прогноз Gradient Boosting"): _png(
            viz.shap_beeswarm(expl)
        ),
        num.figure(SHAP_BAR_CAPTION): _png(viz.shap_bar(expl)),
    }


def _model_figures(metrics: dict, preds: pd.DataFrame, num: Counter) -> dict[str, bytes]:
    curves = {
        k: (*roc_points(preds["t2d"], preds[f"p_{k}"]), m["auc"])
        for k, m in metrics["test"].items()
    }
    figs = {
        num.figure("ROC-криві трьох моделей на тестовій вибірці"): _png(viz.roc_curves(curves)),
        num.figure("Калібрувальні криві на тестовій вибірці"): _png(
            viz.calibration_curves(calibration_frames(metrics))
        ),
    }
    for key in metrics["test"]:
        name = MODEL_NAMES_UK[key]
        fig = viz.confusion_matrix_fig(*confusion_counts(metrics, key), name)
        figs[num.figure(f"Матриця помилок: {name}, тестова вибірка")] = _png(fig)
    return figs


def _ppv_pct(metrics: dict, prev_pct: float) -> float:
    """PPV основної моделі (%) за її чутливості й специфічності на тесті при поширеності prev."""
    t = metrics["test"][metrics["best_model"]]
    return 100 * ppv_at_prevalence(t["recall"], t["specificity"], prev_pct / 100)


def _delong_numbers(metrics: dict) -> dict[str, Number]:
    """Сирі й скориговані за Холмом p попарних порівнянь моделей: p_[holm_]delong_<a>_<b>."""
    numbers: dict[str, Number] = {}
    for r, p_holm in zip(metrics["delong"], delong_holm(metrics), strict=True):
        pair = f"{r['a']}_{r['b']}"
        numbers |= {f"p_delong_{pair}": float(r["p"]), f"p_holm_delong_{pair}": float(p_holm)}
    numbers["n_delong_holm_significant"] = sum(p < ALPHA for p in delong_holm(metrics))
    return numbers


def _model_numbers(metrics: dict) -> dict[str, Number]:
    cv = {f"cv_auc_{k}": round(v["cv_auc_mean"], 3) for k, v in metrics["cv"].items()}
    test = {f"test_auc_{k}": round(v["auc"], 3) for k, v in metrics["test"].items()}
    return {
        "best_model": metrics["best_model"],
        "best_model_rule": metrics.get("best_model_rule", ""),
        **cv,
        **test,
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "best_model_conclusion": best_model_conclusion(metrics),
        **_delong_numbers(metrics),
    }


def _model_tables(train: pd.DataFrame, metrics: dict, num: Counter) -> Part:
    coefs = odds.multivariable_logit(train, FEATURES)
    summary = metrics_summary_table(metrics).rename_axis("Модель").reset_index()
    tables = {
        num.table("Коефіцієнти логістичної регресії на навчальній вибірці"): coef_view(coefs),
        num.table("Метрики трьох моделей: крос-валідація на навчальній частині та тест"): summary,
        num.table("Попарне порівняння AUC моделей (тест ДеЛонга, тестова вибірка)"): delong_table(
            metrics
        ),
        num.table("Пороги Юдена та відповідні чутливість і специфічність"): thresholds_table(
            metrics
        ),
        num.table("Довірчі інтервали чутливості, специфічності, PPV і калібрування"): (
            metrics_ci_table(metrics)
        ),
    }
    return tables, {}, {"n_logit_train": int(coefs.attrs["n"])}


def section_models(
    cohort: pd.DataFrame, models: dict, metrics: dict, preds: pd.DataFrame, num: Counter
) -> Part:
    """Сторінка 4: коефіцієнти logit (train), метрики, ДеЛонг, пороги, ДІ; ROC/калібрування/SHAP."""
    from risk.models.explain import global_shap, make_explainer  # shap важкий: лише тут

    test_rows = cohort["SEQN"].isin(preds["SEQN"])
    tables, _, numbers = _model_tables(cohort[~test_rows], metrics, num)
    expl = global_shap(make_explainer(models[SHAP_MODEL]), cohort[test_rows][FEATURES])
    figs = _model_figures(metrics, preds, num) | _shap_figures(expl, num)
    weighted = float(prevalence(cohort).iloc[0]["weighted_pct"])
    numbers |= _model_numbers(metrics) | {
        "shap_top3": _shap_top3(expl),
        "ppv_at_us_prev": _ppv_pct(metrics, weighted),
    }
    return tables, figs, numbers


# ---------- 5. FINDRISC (сторінка 5) ----------


def findrisc_delong_texts(metrics: dict) -> dict[str, str]:
    """Варіант FINDRISC -> речення ДеЛонга «основна модель проти анкети» з p (Холм)."""
    rows = zip(metrics["delong_vs_findrisc"], findrisc_holm(metrics), strict=True)
    return {
        r["variant"]: interpret.delong_text(
            MODEL_NAMES_UK[r["model"]], r["auc_a"], f"FINDRISC ({r['variant']})", r["auc_b"],
            r["p"], p_holm=p_holm,
        )
        for r, p_holm in rows
    }  # fmt: skip


def _findrisc_numbers(metrics: dict, preds: pd.DataFrame) -> dict[str, Number]:
    numbers: dict[str, Number] = {
        f"findrisc_auc_{k}": round(v["auc"], 3) for k, v in metrics["findrisc"].items()
    }
    for v, text in findrisc_delong_texts(metrics).items():
        numbers[f"delong_vs_findrisc_{v}"] = text
    for r, p_holm in zip(metrics["delong_vs_findrisc"], findrisc_holm(metrics), strict=True):
        numbers |= {f"p_delong_vs_findrisc_{r['variant']}": float(r["p"]),
                    f"p_holm_delong_vs_findrisc_{r['variant']}": float(p_holm)}  # fmt: skip
    excluded = preds.loc[preds["findrisc_a"].isna(), "t2d"]
    numbers["frd_a_excluded_n"] = len(excluded)
    numbers["frd_a_excluded_pos_pct"] = float(excluded.mean() * 100) if len(excluded) else 0.0
    return numbers


def _findrisc_figures(v: FindriscVariant, num: Counter) -> dict[str, bytes]:
    return {
        num.figure("Розподіл балів FINDRISC (варіант a) у групах без ЦД2 і з ЦД2"): _png(
            viz.findrisc_distribution(v.score, v.y)
        ),
        num.figure("Частка людей з ЦД2 за категоріями FINDRISC (варіант a)"): _png(
            viz.findrisc_categories(categories(v.score), v.y, max_score=MAX_SCORE_A)
        ),
    }


def cohort_findrisc(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Number]]:
    """AUC варіантів (a) і (b) на всій аналітичній вибірці — головна оцінка шкали.

    FINDRISC не навчається на даних, тому відкладена вибірка йому не потрібна; з моделлю
    анкету порівнюють лише на тестовій вибірці (таблиця 5.4).
    """
    rows: list[dict[str, Number]] = []
    numbers: dict[str, Number] = {}
    for key, v in variants(cohort).items():
        if key not in COHORT_FINDRISC:
            continue
        auc, lo, hi = auc_ci(v.y.to_numpy(), v.score.to_numpy())
        ci = f"{lo:.3f}–{hi:.3f}"
        rows.append({"Варіант": COHORT_FINDRISC[key], "N": len(v.y), "Позитивних": int(v.y.sum()),
                     "AUC": round(auc, 3), "95% ДІ (ДеЛонга)": ci})  # fmt: skip
        numbers |= {f"frd_cohort_auc_{key}": f"{auc:.3f} ({ci})", f"frd_cohort_n_{key}": len(v.y),
                    f"frd_cohort_pos_{key}": int(v.y.sum())}  # fmt: skip
    return pd.DataFrame(rows), numbers


def _findrisc_tables(
    v: FindriscVariant, metrics: dict, cohort_auc: pd.DataFrame, sub: pd.DataFrame, num: Counter
) -> dict[str, pd.DataFrame]:
    """Таблиці 5.1–5.6 у порядку сторінки FINDRISC."""
    cats = category_table(v.score, v.y, max_score=MAX_SCORE_A).round(1)
    return {
        num.table("Оригінальний і модифікований FINDRISC"): pd.DataFrame(
            ORIGINAL_VS_MODIFIED
        ).rename(columns=ORIGINAL_COLS_UK),
        num.table(
            "Розподіл за категоріями ризику FINDRISC (варіант (a), уся аналітична вибірка; "
            f"максимум {MAX_SCORE_A} балів, тому категорія «дуже високий» недосяжна)"
        ): cats.rename(columns=CATEGORY_COLS_UK),
        num.table("AUC трьох варіантів FINDRISC на тестовій вибірці"): findrisc_auc_table(metrics),
        num.table("Порівняння основної моделі з FINDRISC (тест ДеЛонга, тестова вибірка)"): (
            findrisc_vs_model_table(metrics)
        ),
        num.table("AUC FINDRISC на всій аналітичній вибірці (описово)"): cohort_auc,
        num.table(SUBGROUP_CAPTION): subgroup_view(sub, metrics["best_model"]),
    }


def section_findrisc(
    cohort: pd.DataFrame, metrics: dict, preds: pd.DataFrame, num: Counter
) -> Part:
    """Сторінка 5: бали, категорії (варіант a), AUC на тесті й на всій вибірці, підгрупи."""
    v = variants(cohort)["a"]
    cohort_auc, cohort_numbers = cohort_findrisc(cohort)
    sub = subgroup_auc(preds, metrics["best_model"])
    tables = _findrisc_tables(v, metrics, cohort_auc, sub, num)
    figs = _findrisc_figures(v, num)
    numbers = _findrisc_numbers(metrics, preds) | cohort_numbers | _subgroup_numbers(sub, metrics)
    return tables, figs, numbers


def _subgroup_numbers(sub: pd.DataFrame, metrics: dict) -> dict[str, Number]:
    best = metrics["best_model"]
    return {
        **subgroup_numbers(sub),
        "sub_pct_diagnosed_test": pct_diagnosed(sub, best),
        "subgroup_conclusion": subgroup_conclusion(sub, best),
    }


# ---------- 6. Порівняння з Україною (сторінка 6) ----------


def nhanes_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    """Рядки цієї роботи для таблиці 6.1, зважені, як оцінки NCHS.

    ЦД2 загалом (усі дорослі і 20–79 років, як у IDF), окремо діагностований
    і недіагностований.
    """
    years = [k.split("-") for k in CYCLES]
    period = f"{years[0][0]}–{years[-1][-1]}"
    status, adults, own = with_status(cohort), f"≥ {ADULT_AGE} років", "США (NHANES, ця робота)"
    groups = [
        (MEASURE_TOTAL, own, adults, NHANES_METHOD, cohort, "t2d"),
        (MEASURE_TOTAL, "США (NHANES)", f"{ADULT_AGE}–{AGE_TOPCODE - 1} років",
         f"{NHANES_METHOD}, сира зважена", cohort[cohort["age"] < AGE_TOPCODE], "t2d"),
        (MEASURE_DIAG, own, adults, "самозвіт про діагноз, зважена", status, "dm_diag"),
        (MEASURE_UNDIAG, own, adults, f"HbA1c ≥ {HBA1C_THRESHOLD} % без діагнозу, зважена",
         status, "dm_undiag"),
    ]  # fmt: skip
    rows = []
    for measure, source, age, method, data, outcome in groups:
        p = prevalence(data, outcome).iloc[0]
        rows.append([measure, source, period, age, method, p["weighted_pct"], p["ci_low"],
                     p["ci_high"], "—"])  # fmt: skip
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def comparison_view(comparison: pd.DataFrame) -> pd.DataFrame:
    """Таблиця 6.1 українською: поширеність до 2 знаків, «—» там, де інтервалу немає."""
    ci = {c: comparison[c].map(lambda x: interpret.fmt_cell(x, 2)) for c in ("ci_low", "ci_high")}
    view = comparison.assign(value=comparison["value"].round(2), **ci)
    return view.rename(columns=COMPARISON_COLS_UK)


def dots_data(table: pd.DataFrame) -> pd.DataFrame:
    """Підпис рядка рисунка: джерело і що вимірює, а під ним метод (коротко) і вік."""
    method = table["method"].str.replace(f"; {CALC_NOTE}", ", розрахунок", regex=False)
    label = (
        table["source"] + ": " + table["measure"].str.lower() + "\n"
        + method + ", " + table["age"]
    )  # fmt: skip
    return table.assign(label=label)[["label", "value", "ci_low", "ci_high"]]


def section_ukraine(
    cohort: pd.DataFrame, ukraine: pd.DataFrame, us: pd.DataFrame, metrics: dict, num: Counter
) -> Part:
    """Сторінка 6: поширеність діабету в США і Україні; без даних по Україні секція порожня."""
    if ukraine.empty:
        return {}, {}, {}
    comparison = prevalence_comparison(nhanes_rows(cohort), ukraine, us)
    tables = {num.table(COMPARISON_CAPTION): comparison_view(comparison)}
    figs = {num.figure(COMPARISON_CAPTION): _png(viz.prevalence_dots(dots_data(comparison)))}
    tables[num.table("Інші показники")] = other_indicators(ukraine, us).rename(
        columns=OTHER_COLS_UK
    )
    return tables, figs, _ukraine_numbers(ukraine, metrics)


def ppv_basis_text(crude_pct: float) -> str:
    """Пояснення, від якої поширеності рахується PPV для України."""
    return (
        f"PPV для України розраховано від сирої поширеності {crude_pct:.1f} % (кількість "
        f"дорослих 20–79 з діабетом / доросле населення 20–79 за IDF; {CALC_NOTE}), а не від "
        "стандартизованої за віком: PPV залежить від фактичної частки хворих серед населення."
    )


def _ukraine_numbers(ukraine: pd.DataFrame, metrics: dict) -> dict[str, Number]:
    """Частки з кількостей IDF і МОЗ, МОЗ проти IDF для діагностованих, PPV від сирої частки."""
    shares = ukraine_shares(ukraine)
    numbers: dict[str, Number] = {f"ua_prev_{k}": v for k, v in shares.items()}
    ratio, text = moz_vs_idf(ukraine), moz_idf_text(ukraine)
    if ratio is not None and text is not None:
        counts = ukraine_counts(ukraine)
        numbers |= {"ua_idf_diag_n": int(counts["diag"][0]), "ua_moz_n": int(counts["moz"][0]),
                    "ua_moz_vs_idf_pct": ratio, "ua_moz_idf_text": text}  # fmt: skip
    if "crude" in shares:
        numbers["ppv_at_ua_prev"] = _ppv_pct(metrics, shares["crude"])
        numbers["ppv_at_ua_prev_basis"] = ppv_basis_text(shares["crude"])
    return numbers


# ---------- складання ----------


def build_results(
    cohort: pd.DataFrame,
    flow: list[FlowStep],
    variables: pd.DataFrame,
    *,
    models: dict,
    metrics: dict,
    preds: pd.DataFrame,
    ukraine: pd.DataFrame,
    us_reference: pd.DataFrame | None = None,
) -> ResultsBundle:
    """Усі секції; нумерація — за сторінками застосунку (1.x … 6.x).

    us_reference — оцінки NCHS для США; за замовчуванням читаються з data/external.
    """
    us = load_us_reference() if us_reference is None else us_reference
    n1, n2, n3, n4, n5, n6 = (Counter(i) for i in range(1, 7))
    parts = [
        section_sample(cohort, flow, variables, n1),
        section_descriptive(cohort, n2),
        section_correlations(cohort, n2),
        section_prevalence(cohort, n2),
        section_stats_extra(cohort, n2),
        section_hypotheses(cohort, n3),
        section_odds(cohort, n3),
        section_models(cohort, models, metrics, preds, n4),
        section_findrisc(cohort, metrics, preds, n5),
        section_ukraine(cohort, ukraine, us, metrics, n6),
    ]
    return ResultsBundle(*_merge(parts), glossary_frame())


def _merge(parts: list[Part]) -> Part:
    """Таблиці, рисунки й числа всіх секцій разом, у порядку секцій."""
    tables: dict[str, pd.DataFrame] = {}
    figs: dict[str, bytes] = {}
    numbers: dict[str, Number] = {}
    for t, f, n in parts:
        tables |= t
        figs |= f
        numbers |= n
    return tables, figs, numbers


# ---------- експорт ----------


def _sheet_name(caption: str) -> str:
    number = caption.removeprefix("Таблиця ").split(". ", 1)[0]
    return SHEET_NAMES.get(number, f"{number} {caption.split('. ', 1)[1]}"[:31])


def to_excel_sheets(bundle: ResultsBundle) -> list[ExcelSheet]:
    """По одному листу на таблицю; повний підпис — приміткою над таблицею."""
    return [
        ExcelSheet(_sheet_name(k), v, note=k, float_format=CORR_FORMAT if "кореляц" in k else None)
        for k, v in bundle.tables.items()
    ]


def export_excel(bundle: ResultsBundle) -> bytes:
    """Книга .xlsx: лист на таблицю, «Графіки» і «Пояснення» (байти)."""
    return build_workbook(to_excel_sheets(bundle), bundle.figures, bundle.glossary)


def zip_caption(caption: str) -> str:
    """Підпис для імені файлу: без позначки «лише в експорті», «:» → «—» (а не «-_»)."""
    return caption.replace(ONLY_EXPORT, "").replace(": ", " — ")


def export_zip(bundle: ResultsBundle) -> bytes:
    """ZIP з усіма рисунками у форматі PNG (байти)."""
    return figures_zip({zip_caption(k): v for k, v in bundle.figures.items()})
