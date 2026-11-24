"""Сторінка 1: джерело даних, відбір учасників, змінні, завантаження власних даних."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from _common import (
    Numbering,
    accept_user_file,
    active_dataset,
    dataset_banner,
    disclaimer,
    figure_block,
    fmt_int,
    load_flow,
    load_variables,
    on_upload_change,
    page_header,
    prevalence_table,
    setup_page,
    table_block,
    uploader_key,
)
from risk.data.preprocessing import preprocessing_table
from risk.data.user_data import (
    MIN_PER_CLASS,
    MIN_ROWS,
    template_csv,
    template_description,
    template_xlsx,
)
from risk.report.results import flow_table, prevalence_view
from risk.viz import figures

setup_page("Дані і вибірка", "📊")
page_header(
    "Дані і вибірка",
    "Тут показано, звідки взято дані і як з усіх учасників опитування ми відібрали тих, "
    "кого можна аналізувати. Також видно, які ознаки людини використовуються і скільки в них "
    "пропусків. Внизу можна завантажити власний файл, і тоді статистика перерахується на ньому.",
)

df, is_user = active_dataset()
dataset_banner(is_user, len(df))
num = Numbering(1)
# номери таблиць задано явно: 1.1–1.4 збігаються з аркушами Excel, 1.5 (шаблон) — лише тут
T_STEPS, T_VARIABLES, T_PREVALENCE, T_PREPROCESSING, T_TEMPLATE = (f"1.{i}" for i in range(1, 6))

st.header("Джерело даних")
st.markdown(
    """
Дані взято з національного опитування здоров'я та харчування **NHANES** (Центри з контролю
і профілактики захворювань США, CDC): [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/).
Використано два цикли: **2015–2016** та **2017 – березень 2020**.

Діабет визначаємо за рівнем **HbA1c** (глікований гемоглобін, середній цукор крові за
2–3 місяці) та самозвітом про діагноз. HbA1c обрано, бо це лабораторний вимір, а не відповідь
людини: так ми знаходимо й тих, хто ще не знає про свій діабет.
"""
)

st.header("Відбір учасників")
if is_user:
    st.info("Блок-схема відбору стосується лише даних NHANES; для вашого файлу вона не будується.")
else:
    figure_block(
        figures.flowchart,
        (load_flow(),),
        num.figure(),
        "Блок-схема відбору учасників до аналітичної вибірки",
        "Зверху вниз: кожен блок показує, скільки людей лишилось після чергового кроку; "
        "праворуч, скільки виключено на цьому кроці.",
        "Аналізуються лише дорослі, не вагітні, з виміряним HbA1c і визначеним статусом "
        "діабету. Так результати не спотворені відсутніми аналізами.",
    )
    table_block(
        flow_table(load_flow()),
        T_STEPS,
        "Кроки відбору учасників",
        "Ті самі кроки, що на блок-схемі: скільки людей було на початку кожного кроку "
        "і скільки виключено.",
        "За таблицею видно, на якому кроці відсіялось найбільше людей.",
    )

st.header("Змінні")
variables = load_variables().rename(
    columns={
        "variable": "Змінна",
        "label_uk": "Назва",
        "nhanes_code": "Код NHANES",
        "unit": "Одиниці",
        "missing_pct": "Пропуски, %",
    }
)
table_block(
    variables,
    T_VARIABLES,
    "Змінні дослідження",
    "Кожен рядок — це одна ознака людини: її назва, код у NHANES, одиниці виміру і частка "
    "відсутніх значень.",
    "Чим більше пропусків, тим менше людей можна використати для цієї ознаки. "
    "Це враховується в розрахунках і окремо вказується в таблицях.",
    wrap=True,
)

st.header("Розподіл діабету 2 типу в аналітичній вибірці")
if is_user:
    st.info("Зважена поширеність розраховується лише для даних NHANES (потрібні ваги опитування).")
else:
    prev = prevalence_table(df)
    by_cycle = prev[prev["group"] != "Разом"].reset_index(drop=True)
    table_block(
        prevalence_view(by_cycle),
        T_PREVALENCE,
        "Кількість учасників і частка ЦД2 за циклами",
        "Для кожного циклу: скільки людей, яка частка з діабетом у самій вибірці "
        "(незважена) і з урахуванням ваг опитування (зважена, з довірчим інтервалом).",
        "Незважена частка описує вибірку, зважена ближча до населення США. "
        "Для порівняння груп у вибірці використовуємо незважені дані.",
    )
    figure_block(
        figures.prevalence_bars,
        (by_cycle,),
        num.figure(),
        "Частка ЦД2 за циклами: незважена та зважена оцінки",
        "Сірий стовпчик — це частка у вибірці, синій — це зважена оцінка з довірчим "
        "інтервалом (вусики).",
        "Якщо вусики двох циклів перекриваються, істотної різниці між циклами немає.",
    )

st.header("Попередня обробка даних")
table_block(
    preprocessing_table(),
    T_PREPROCESSING,
    "Попередня обробка даних",
    "Кожен рядок — один крок підготовки: що саме робимо з даними і на якому етапі.",
    "Викиди не видаляються, пропуски в моделях заповнюються медіаною навчальної частини "
    "(Gradient Boosting працює з пропусками напряму). Усе це робиться до оцінювання моделей, "
    "а не на тестових людях.",
    wrap=True,
)

st.header("Завантажити власні дані")
st.markdown(
    "Завантажте шаблон, заповніть його своїми даними й завантажте назад. "
    f"У шаблоні лише приклади, у вашому файлі має бути щонайменше {MIN_ROWS} рядків, "
    f"а людей з діабетом і без нього — не менше ніж по {MIN_PER_CLASS}. "
    "Після цього сторінки **Статистика** та **Фактори ризику** будуть розраховані на вашому файлі "
    "(докладніше про те, що перераховується, нижче)."
)
csv_col, xlsx_col = st.columns(2)
csv_col.download_button(
    "Завантажити шаблон CSV",
    data=template_csv(),
    file_name="шаблон.csv",
    mime="text/csv",
    key="dl_template",
)
xlsx_col.download_button(
    "Завантажити шаблон Excel (.xlsx)",
    data=template_xlsx(),
    file_name="шаблон.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="dl_template_xlsx",
)
table_block(
    template_description(),
    T_TEMPLATE,
    "Опис колонок шаблону для власного файлу",
    "Кожен рядок описує одну колонку шаблону: назву, допустимий діапазон і чи вона обов'язкова.",
    "Обов'язкові колонки мають бути заповнені в усіх рядках, решту можна залишити порожніми.",
)

# без type=: інакше браузер показує англійське повідомлення замість нашого
uploaded = st.file_uploader(
    "Ваш файл (CSV або Excel)", key=uploader_key(), on_change=on_upload_change
)
if uploaded is not None:
    if uploaded.file_id != st.session_state.get("user_file_id"):
        accept_user_file(uploaded.getvalue(), uploaded.name, uploaded.file_id)
        st.rerun()  # щоб уся сторінка відобразила новий стан
    for err in st.session_state.get("user_errors", []):
        st.error(err)
    if "user_df" in st.session_state:
        st.success(
            f"Файл прийнято: N = {fmt_int(len(st.session_state['user_df']))}. "
            "Дані збережено для всіх сторінок."
        )
st.caption(
    f"Завеликий файл (понад {st.get_option('server.maxUploadSize')} МБ) буде позначено "
    "червоним. Старий формат .xls не підтримується — збережіть як .xlsx."
)

st.markdown(
    """
**Що перераховується на вашому файлі:** сторінки «Статистика» та «Фактори ризику»
(описова статистика, перевірка гіпотез, відношення шансів, кореляції), а також сторінка
«FINDRISC» — варіант (a), якщо у файлі є колонки pa_h_week, bp_med і fam_dm.

**Що лишається за NHANES:** блок-схема відбору, зважена поширеність, порівняння з Україною
і навчання моделей. Моделі не перенавчаються, але сторінка «Моделі» показує AUC збережених
моделей на вашому файлі.
"""
)

disclaimer()
