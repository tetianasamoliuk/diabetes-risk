"""Сторінка 5: модифікований FINDRISC, його розподіл, AUC і порівняння з моделлю."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from _common import (
    Numbering,
    active_dataset,
    dataset_banner,
    disclaimer,
    figure_block,
    fmt_int,
    load_metrics,
    load_predictions,
    models_or_stop,
    page_header,
    setup_page,
    table_block,
)
from risk import interpret
from risk.config import FEATURE_LABELS_UK, MODEL_NAMES_UK
from risk.findrisc.score import (
    MAX_SCORE_A,
    ORIGINAL_VS_MODIFIED,
    categories,
    category_table,
    variants,
)
from risk.models.delong import auc_ci
from risk.models.summary import (
    SUBGROUP_CAPTION,
    findrisc_auc_table,
    findrisc_vs_model_table,
    subgroup_auc,
    subgroup_comparisons,
    subgroup_conclusion,
    subgroup_view,
)
from risk.report.results import (
    CATEGORY_COLS_UK,
    ORIGINAL_COLS_UK,
    cohort_findrisc,
    findrisc_delong_texts,
)
from risk.viz import figures

# Колонки, потрібні для балу варіанта (a)
FINDRISC_A_COLUMNS = ["age", "male", "bmi", "waist", "pa_h_week", "bp_med", "fam_dm"]


@st.cache_data(show_spinner="Обчислення…")
def compute_variant_a(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Бали й справжній статус для варіанта (a) на активному наборі."""
    v = variants(df)["a"]
    return v.score, v.y


def no_findrisc_message(data: pd.DataFrame) -> str:
    """Одне пояснення, чому бал FINDRISC не порахувати: яких колонок бракує і що в них."""
    missing = [c for c in FINDRISC_A_COLUMNS if c not in data.columns or data[c].isna().all()]
    head = "FINDRISC для вашого файлу порахувати не можна: бал рахується лише для людей, "
    if not missing:
        return head + "у яких заповнені всі пункти анкети, а таких рядків у файлі немає."
    cols = "; ".join(f"{c} — {FEATURE_LABELS_UK[c]}" for c in missing)
    return (
        f"{head}у яких заповнені всі пункти анкети, а у файлі бракує колонок: {cols}. "
        "Додайте їх за шаблоном зі сторінки «Дані і вибірка»."
    )


@st.cache_data(show_spinner="Обчислення…")
def whole_sample_auc(data: pd.DataFrame) -> pd.DataFrame:
    """Таблиця AUC варіантів (a) і (b) на всій аналітичній вибірці."""
    return cohort_findrisc(data)[0]


def exclusion_note(metrics: dict) -> str:
    """Чому N у порівнянні з FINDRISC (a) менший за тест на сторінці «Моделі»."""
    row = next(r for r in metrics["delong_vs_findrisc"] if r["variant"] == "a")
    return (
        f"AUC моделі тут рахується на {fmt_int(row['n'])} людях, а не на всіх "
        f"{fmt_int(metrics['n_test'])} "
        f"тестових (сторінка «Моделі»): {fmt_int(metrics['n_test'] - row['n'])} людей без повних "
        "відповідей анкети виключено, бо порівнювати можна лише на тих самих людях."
    )


setup_page("FINDRISC", "📋")
page_header(
    "FINDRISC",
    "FINDRISC — відома анкета з кількох питань (вік, вага, талія, активність, тиск, "
    "спадковість), яку фінські лікарі розробили, щоб швидко оцінити ризик діабету 2 типу "
    "без аналізів. Ми використовуємо її модифіковану версію, бо в NHANES немає питання "
    "про овочі/фрукти, а спадковість охоплює лише найближчих родичів. Нижче видно, що "
    "змінено, як бали розподілені серед людей і чи краща за анкету машинна модель.",
)

df, is_user = active_dataset()
dataset_banner(is_user, len(df))
num = Numbering(5)

st.header("Бали: оригінал і модифікація")
original_view = pd.DataFrame(ORIGINAL_VS_MODIFIED)
table_block(
    original_view.rename(columns=ORIGINAL_COLS_UK),
    num.table(),
    "Оригінальний і модифікований FINDRISC",
    "Кожен рядок — одне питання анкети і скільки балів воно дає. Останній стовпчик "
    "пояснює, що ми змінили.",
    "Сума балів показує ризик: що більше балів, то вищий ризик. У нашій версії максимум "
    "25 балів (20 без пункту про високу глюкозу в анамнезі).",
    wrap=True,
)

st.header("Розподіл балів")
score, y = compute_variant_a(df)
if score.notna().sum() == 0:
    st.warning(no_findrisc_message(df))
    disclaimer()
    st.stop()
if is_user:
    st.info(
        "У шаблоні файлу немає пункту про високу глюкозу в анамнезі, тому рахується "
        "варіант (a): шість пунктів анкети без нього, максимум 20 балів."
    )
figure_block(
    figures.findrisc_distribution(score, y),
    num.figure(),
    "Розподіл балів FINDRISC (варіант a) у групах без ЦД2 і з ЦД2",
    "По горизонталі — сума балів, по вертикалі — кількість людей. Окремі стовпчики для "
    "людей без діабету та з діабетом.",
    "Якщо стовпчики хворих зсунуті праворуч, анкета розрізняє групи: у хворих балів більше.",
)
cat_view = category_table(score, y, max_score=MAX_SCORE_A).round(1).rename(columns=CATEGORY_COLS_UK)
table_block(
    cat_view,
    num.table(),
    "Розподіл за категоріями ризику FINDRISC (варіант (a), усі учасники; максимум "
    "20 балів, тому категорія «дуже високий» недосяжна)",
    "Бали згруповано у чотири досяжні категорії з п'яти. Для кожної показано, скільки людей "
    "у ній і яка частка з них має діабет 2 типу.",
    "Частка хворих має зростати від низької категорії до високої. Якщо так, "
    "категорії працюють як грубий індикатор ризику.",
)
st.caption(
    f"Для варіанта (a) максимум {MAX_SCORE_A} балів, тому категорія «дуже високий» (21–25) "
    "недосяжна."
)
figure_block(
    figures.findrisc_categories(categories(score), y, max_score=MAX_SCORE_A),
    num.figure(),
    "Частка людей з ЦД2 за категоріями FINDRISC (варіант a)",
    "Висота стовпчика — відсоток людей з діабетом у категорії, під стовпчиком кількість людей.",
    "Зростання стовпчиків зліва направо означає, що вища категорія справді відповідає "
    "вищому ризику.",
)

if is_user:
    st.info("AUC варіантів і порівняння з моделлю доступні лише для даних NHANES.")
    disclaimer()
    st.stop()

models_or_stop()
metrics = load_metrics()
best = metrics["best_model"]

st.header("Наскільки добре анкета розрізняє людей з ЦД2")
whole = whole_sample_auc(df)
auc_a, lo_a, hi_a = auc_ci(y.to_numpy(), score.to_numpy())  # variants() уже без пропусків
st.markdown(
    f"**Уся аналітична вибірка, особи з повною анкетою, варіант (a), N = {fmt_int(len(y))}.** "
    + interpret.auc_text(auc_a, lo_a, hi_a)
    + " Це головна оцінка шкали: FINDRISC не навчається на даних, тому відкладена тестова "
    "вибірка йому не потрібна, а оцінка на всіх людях з повною анкетою точніша. Обидва "
    "варіанти на всій вибірці — у таблиці «AUC FINDRISC на всій аналітичній вибірці» нижче."
)

st.header("AUC FINDRISC на тестовій вибірці")
st.write(
    "Модель оцінюють лише на тестовій вибірці, якої вона не бачила під час навчання. Щоб "
    "порівняти анкету з моделлю чесно, на тих самих людях, AUC анкети тут теж рахуємо на "
    "тестовій вибірці."
)
table_block(
    findrisc_auc_table(metrics),
    num.table(),
    "AUC трьох варіантів FINDRISC на тестовій вибірці",
    "Варіанти відрізняються тим, як врахований пункт про високу глюкозу в анамнезі. "
    "(a) без цього пункту; (b) з пунктом, але лише для людей без діагнозу; (c) з пунктом "
    "для всіх.",
    "Варіант (c) некоректний: людям з уже встановленим діабетом ми приписуємо відповідь "
    "«так» на питання, яке їм фактично не ставили. Анкета тоді «вгадує» діагноз, який уже "
    "відомий, і AUC штучно завищений. Його показано лише як демонстрацію витоку.",
    wrap=True,
)
fa = metrics["findrisc"]["a"]
fa_text = interpret.auc_text(fa["auc"], fa["ci_low"], fa["ci_high"])
st.markdown(f"**Варіант (a), тестова вибірка.** {fa_text}")

st.header(f"Порівняння з основною моделлю ({MODEL_NAMES_UK[best]})")
table_block(
    findrisc_vs_model_table(metrics),
    num.table(),
    "Порівняння основної моделі з FINDRISC (тест ДеЛонга, тестова вибірка)",
    "Для кожного варіанта FINDRISC перевіряється, чи AUC моделі статистично відрізняється "
    "від AUC анкети на одних і тих самих людях. Порівнянь два, тому p скориговано поправкою "
    "Холма; висновок зроблено за p (Холм).",
    "Значуща різниця означає, що на цих людях модель розрізняє хворих і здорових краще за "
    "анкету, і це не випадковість. Але порівняння не на рівних: модель бачить виміряні тиск "
    "і холестерин і навчена на цій самій популяції (NHANES), а анкета — лише відповіді на "
    "кілька питань, і створена вона на іншій популяції. Тому перевага моделі частково "
    "пояснюється додатковими вимірюваннями і «домашніми» даними. Порівнювати можна лише "
    "придатні варіанти (a) і (b). " + exclusion_note(metrics),
)
for variant, text in findrisc_delong_texts(metrics).items():
    st.markdown(f"**Варіант ({variant}).** {text}")
fb = metrics["findrisc"]["b"]
st.caption(
    f"Примітка. У варіанті (b) лише {fmt_int(fb['n_pos'])} людей із недіагностованим діабетом "
    f"(із {fmt_int(fb['n'])}), тому його оцінка менш стабільна: довірчий інтервал широкий."
)

st.header("AUC FINDRISC на всій аналітичній вибірці")
table_block(
    whole,
    num.table(),
    "AUC FINDRISC на всій аналітичній вибірці (описово)",
    "AUC варіантів (a) і (b) на всіх людях аналітичної вибірки з повною анкетою, 95% ДІ за "
    "методом ДеЛонга. Варіант (b) — лише люди без встановленого діагнозу.",
    "Це головна оцінка якості анкети. Вона описова: з моделлю тут не порівнюємо, бо модель "
    "бачила більшість цих людей під час навчання.",
)

st.header("Діагностований і недіагностований ЦД2 окремо")
st.write(
    "Більшість людей з ЦД2 у вибірці вже знають свій діагноз, а скринінг потрібен насамперед "
    "для тих, хто не знає. Тому моделі й анкету окремо порівняно на двох підгрупах тестової "
    "вибірки: недіагностований ЦД2 проти осіб без ЦД2 і діагностований ЦД2 проти осіб без ЦД2."
)
subgroups = subgroup_auc(load_predictions(), best)
table_block(
    subgroup_view(subgroups, best),
    num.table(),
    SUBGROUP_CAPTION,
    "Для кожної підгрупи: AUC трьох моделей і FINDRISC (a) з 95% ДІ, а також тест ДеЛонга "
    "«основна модель проти логістичної регресії» і «основна модель проти FINDRISC (a)». "
    "Різницю з анкетою рахуємо лише на людях з повною анкетою (N у рядку FINDRISC), тому "
    "вона може не дорівнювати різниці стовпчика AUC. p (Холм) — з поправкою на чотири "
    "порівняння таблиці.",
    subgroup_conclusion(subgroups, best),
    wrap=True,
)
for line in subgroup_comparisons(subgroups, best):
    st.markdown(f"- {line}")

disclaimer()
