"""Сторінка 4: три моделі, їхня якість, порівняння та SHAP (лише NHANES)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import shap
import streamlit as st

from _common import (
    Numbering,
    disclaimer,
    figure_block,
    fmt_int,
    get_models,
    load_cohort,
    load_metrics,
    load_predictions,
    models_or_stop,
    nhanes_only_banner,
    page_header,
    setup_page,
    table_block,
)
from risk import interpret
from risk.config import (
    CALIBRATION_BINS,
    CV_FOLDS,
    FEATURES,
    MODEL_NAMES_UK,
    TEST_PCT,
    TRAIN_PCT,
)
from risk.models.calibration import calibration_summary
from risk.models.delong import auc_ci
from risk.models.evaluate import roc_points
from risk.models.explain import global_shap, make_explainer
from risk.models.summary import (
    best_model_conclusion,
    calibration_frames,
    confusion_counts,
    delong_table,
    metrics_summary_table,
)
from risk.report.results import SHAP_BAR_CAPTION, coef_view, thresholds_table
from risk.stats.odds import multivariable_logit
from risk.viz import figures


@st.cache_data(show_spinner="Обчислення…")
def train_coefficients() -> pd.DataFrame:
    """Багатофакторний logit на навчальних рядках (вибірка без SEQN з тесту)."""
    cohort = load_cohort()
    train = cohort[~cohort["SEQN"].isin(load_predictions()["SEQN"])]
    t = multivariable_logit(train, FEATURES)
    t.attrs = {"n": t.attrs["n"]}
    return t


@st.cache_resource(show_spinner="Обчислення…")
def shap_explanation() -> shap.Explanation:
    """Глобальні SHAP-значення HGB на тестових рядках."""
    cohort = load_cohort()
    test = cohort[cohort["SEQN"].isin(load_predictions()["SEQN"])]
    explainer = make_explainer(get_models()[0]["hgb"])
    return global_shap(explainer, test[FEATURES])


def user_data_problem(df: pd.DataFrame) -> str | None:
    """Людське пояснення, чому набір непридатний для перевірки моделей (або None)."""
    missing = [c for c in [*FEATURES, "t2d"] if c not in df.columns]
    if missing:
        return "У вашому наборі бракує колонок: " + ", ".join(missing) + "."
    if set(df["t2d"].dropna().astype(int)) != {0, 1}:
        return "У колонці t2d має бути обидва класи (0 і 1), інакше AUC порахувати неможливо."
    return None


def user_models_table(df: pd.DataFrame) -> pd.DataFrame:
    """N, позитивні та AUC з 95% ДІ (ДеЛонг) збережених моделей на користувацькому наборі."""
    df = df.dropna(subset=["t2d"])
    y = df["t2d"].astype(int).to_numpy()
    rows = []
    for key, model in get_models()[0].items():
        auc, lo, hi = auc_ci(y, model.predict_proba(df[FEATURES])[:, 1])
        rows.append(
            {
                "Модель": MODEL_NAMES_UK[key],
                "N": len(y),
                "Позитивних": int(y.sum()),
                "AUC (95% ДІ)": f"{auc:.3f} ({lo:.3f}–{hi:.3f})",
                "_auc": auc,
            }
        )
    table = pd.DataFrame(rows)
    top = table.loc[table["_auc"].idxmax()]
    table = table.drop(columns="_auc")
    table.attrs["top"] = (top["Модель"], top["_auc"])
    return table


setup_page("Моделі", "🧮")
page_header(
    "Моделі",
    f"Уявіть трьох «учнів», які вчилися на {TRAIN_PCT} % людей з даних NHANES, а іспит "
    f"здають на решті {TEST_PCT} %, яких ніколи не бачили. "
    "Так видно, наскільки добре кожен із них передбачає діабет 2 типу в нових людей. "
    "Нижче: як вони влаштовані, наскільки точні, чи чесні їхні ймовірності "
    "і яку з них обрано основною.",
)
nhanes_only_banner(
    "Таблиці, рисунки й моделі нижче побудовано на NHANES; ваш файл використано лише в "
    "останньому блоці сторінки."
)

models_or_stop()
metrics = load_metrics()
preds = load_predictions()
best = metrics["best_model"]
names = MODEL_NAMES_UK
num = Numbering(4)

st.header("Логістична регресія: формула і коефіцієнти")
coefs = train_coefficients()
st.latex(interpret.logit_formula_aligned(coefs["coef"]))
st.caption(
    f"{interpret.LOGIT_LEGEND} Формула призначена для інтерпретації, а не для прогнозу. "
    f"{interpret.LOGIT_SAME_FORM_NOTE}"
)
view = coef_view(coefs)
st.write(
    "Коефіцієнти пораховані на повних випадках навчальної частини "
    f"(N = {fmt_int(coefs.attrs['n'])}); "
    "тестові люди в неї не потрапили. Ця формула — модель для статистичних висновків "
    "(коефіцієнти, OR, p), а не та, що робить прогнози. Прогнозує інша модель: "
    "регуляризована логістична регресія зі стандартизацією ознак та підстановкою "
    "медіани замість пропусків, навчена на всій навчальній частині. Напрям впливу "
    "більшості ознак у них однаковий, але коефіцієнти не тотожні. ІМТ і талія сильно "
    "пов'язані між собою (див. VIF), тому знак ІМТ нестабільний і залежить від способу "
    "навчання."
)
table_block(
    view,
    num.table(),
    "Коефіцієнти логістичної регресії на навчальній вибірці",
    "Коефіцієнт показує, як ознака змінює «логіт» (log-odds) ризику. OR = e^коефіцієнт: "
    "у скільки разів зростають шанси при збільшенні ознаки на 1 одиницю. VIF показує, "
    "наскільки ознака дублює інші.",
    "Додатний коефіцієнт підвищує ризик, від'ємний знижує. Ознаки мають різні одиниці "
    "(роки, см, так/ні), тому величини коефіцієнтів напряму не порівнюють.",
)

st.header("Якість моделей")
table_block(
    metrics_summary_table(metrics),
    num.table(),
    "Метрики трьох моделей: крос-валідація на навчальній частині та тест",
    f"AUC крос-валідації — середня якість у {CV_FOLDS} повторах навчання на навчальній "
    "частині (SD — стандартне відхилення між повторами). AUC (тест) — іспит на людях, яких "
    "модель не бачила. Чутливість — частка хворих, яких модель знайшла, специфічність — "
    "частка здорових, яких вона не сплутала з хворими.",
    "AUC 0.5 означає вгадування, 1.0 — ідеал. Чим ближче оцінка Браєра до 0, тим точніші "
    "ймовірності. Усі метрики на тесті розраховано з порогом Юдена.",
    wrap=True,
)
for key, m in metrics["test"].items():
    st.markdown(
        f"**{names[key]}, тестова вибірка.** "
        + interpret.auc_text(m["auc"], m["auc_ci_low"], m["auc_ci_high"])
    )
bt = metrics["test"][best]
st.markdown(
    f"**Чому поріг не 50 % ({names[best]}).** "
    + interpret.threshold_text(bt["threshold"], bt["recall"], bt["specificity"])
)

st.header("ROC-криві")
curves = {
    k: (*roc_points(preds["t2d"], preds[f"p_{k}"]), m["auc"]) for k, m in metrics["test"].items()
}
figure_block(
    figures.roc_curves(curves),
    num.figure(),
    "ROC-криві трьох моделей на тестовій вибірці",
    "Крива показує, скільки хворих модель знаходить (вертикаль) ціною скількох помилкових "
    "тривог серед здорових (горизонталь). Пунктир — вгадування.",
    "Чим крива ближче до верхнього лівого кута, тим краща модель. Криві близько одна до "
    "одної означають схожу якість.",
)

st.header("Калібрування")
figure_block(
    figures.calibration_curves(calibration_frames(metrics)),
    num.figure(),
    "Калібрувальні криві на тестовій вибірці",
    f"Люди розбиті на {CALIBRATION_BINS} груп за прогнозованим ризиком. По горизонталі — "
    "середній прогноз моделі в групі, по вертикалі — частка людей, які справді хворі.",
    "Якщо точки лежать на пунктирній діагоналі, ймовірностям моделі можна довіряти як "
    "відсоткам. Вище діагоналі — модель занижує ризик, нижче — завищує. Нахил калібрування "
    "показує, чи не надто крайні (нахил менший за 1) або надто стиснуті (більший за 1) прогнози.",
)
for key, m in metrics["test"].items():
    mean_pred, mean_obs = calibration_summary(preds["t2d"], preds[f"p_{key}"])
    text = interpret.calibration_text(m["brier"], mean_pred, mean_obs, m["calib_slope"])
    st.markdown(f"**{names[key]}.** {text}")

st.header("Матриці помилок")
tabs = st.tabs([names[k] for k in metrics["test"]])
for tab, key in zip(tabs, metrics["test"], strict=True):
    with tab:
        figure_block(
            figures.confusion_matrix_fig(*confusion_counts(metrics, key), names[key]),
            num.figure(),
            f"Матриця помилок: {names[key]}, тестова вибірка",
            "Рядки — реальність, стовпці — рішення моделі. На діагоналі правильні відповіді, "
            "поза діагоналлю помилки: хибні тривоги та пропущені хворі.",
            "Для скринінгу важливо мало пропущених хворих (нижній лівий кут), навіть якщо "
            "є зайві тривоги: їх знімає подальше обстеження.",
        )

st.header("Чи відрізняються моделі статистично")
table_block(
    delong_table(metrics),
    num.table(),
    "Попарне порівняння AUC моделей (тест ДеЛонга, тестова вибірка)",
    "Тест ДеЛонга перевіряє, чи різниця AUC між двома моделями на одних і тих самих людях "
    "більша за випадкову. Порівнянь кілька, тому p скориговано поправкою Холма: що більше "
    "перевірок, то легше випадково отримати «значущий» результат. Висновок зроблено за "
    "p (Холм): якщо він не менший за 0.05, різниця незначуща.",
    "Незначуща різниця означає, що за цими даними не можна стверджувати, що одна модель "
    "точніша за іншу: відмінність AUC може бути випадковою.",
    wrap=True,
)
st.info(best_model_conclusion(metrics))
table_block(
    thresholds_table(metrics),
    num.table(),
    "Пороги Юдена та відповідні чутливість і специфічність",
    "Для кожної моделі: поріг ймовірності, вище якого людину відносять до групи ризику, "
    "і чутливість та специфічність при цьому порозі на тестовій вибірці.",
    "Поріг за Юденом максимізує суму чутливості й специфічності.",
    collapsed=True,
)

st.header("Що впливає на прогноз (SHAP)")
expl = shap_explanation()
st.caption(f"Модель: {names['hgb']}, тестова вибірка (N = {fmt_int(len(expl.values))}).")
figure_block(
    figures.shap_beeswarm(expl),
    num.figure(),
    "SHAP-графік (beeswarm): внесок ознак у прогноз Gradient Boosting",
    "Кожна точка — одна людина. Положення по горизонталі — наскільки ознака змінила "
    "прогноз (у log-odds): праворуч підвищує ризик, ліворуч знижує. Колір — значення ознаки: "
    "червоний високе, синій низьке. Ознаки впорядковані за важливістю.",
    "Якщо червоні точки праворуч, високі значення ознаки підвищують ризик.",
)
figure_block(
    figures.shap_bar(expl),
    num.figure(),
    SHAP_BAR_CAPTION,
    "Довжина стовпчика — середній абсолютний внесок ознаки у прогноз (у log-odds), "
    "незалежно від напрямку.",
    "Довші стовпчики — ознаки, від яких прогноз залежить найбільше. Це не доказ причини, "
    "а опис того, як працює модель.",
)

user_df = st.session_state.get("user_df")
if user_df is not None:
    st.header("Збережені моделі на ваших даних")
    st.write(
        "Моделі навчені на NHANES і тут не перенавчаються: це перевірка збережених моделей "
        "на вашому наборі. Для необов'язкових колонок, яких у файлі немає, підставлено "
        "медіани навчальної вибірки. Якщо у файлі є рядки з NHANES, ці люди вже були у "
        "навчанні, тому AUC завищений."
    )
    problem = user_data_problem(user_df)
    if problem:
        st.info(problem)
    else:
        user_table = user_models_table(user_df)
        table_block(
            user_table,
            num.table(),
            "Збережені моделі на вашому наборі",
            "Для кожної моделі: скільки людей у вашому наборі, скільки з них позитивних (t2d = 1) "
            "і AUC з 95% довірчим інтервалом (метод ДеЛонга).",
            "AUC 0.5 означає вгадування, 1.0 — ідеал. Малий набір дає широкий інтервал.",
        )
        top_name, top_auc = user_table.attrs["top"]
        st.write(
            f"Найвищий AUC на вашому наборі: {top_name} ({top_auc:.3f}). Це не зміна основної "
            f"моделі: основною лишається {names[best]}, її обрано за середнім AUC "
            "крос-валідації на навчальній частині NHANES; тестова вибірка використана лише "
            "для оцінки."
        )

disclaimer()
