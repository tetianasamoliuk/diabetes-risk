"""Спільні функції Streamlit-шару: завантаження даних, блоки рисунків і таблиць.

Тут немає бізнес-логіки: лише виклики src/risk і показ.
"""

import json
import re
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

# сторінки з app/pages/ імпортують _common; src/ — запасний шлях, якщо пакет risk не встановлено
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from risk.config import AUTHOR, DISCLAIMER, PROCESSED_DIR
from risk.data.cohort import FlowStep
from risk.data.user_data import validate_user_file
from risk.errors import ModelNotTrainedError
from risk.models.train import load_models, load_test_predictions
from risk.report.results import prevalence_all
from risk.viz.style import GROUP_LABELS, apply_style, fig_to_png

SURVEY_COLS = ("wt_mec", "strata", "psu")
NO_VARIATION_MSG = "У файлі немає ознак з різними значеннями для цієї таблиці"

# Вбудовані англійські написи Streamlit, які видно постійно: англійський текст сховано,
# а ::after показує український. Кнопка лишається кнопкою (миша, клавіатура), і читач
# екрана називає її українською: тому в неї display: none на тексті, а не visibility.
# Підказка до червоної позначки файла лежить у кінці body, поза завантажувачем, тому її
# знаходить :has() за позначкою з помилкою й розміром у МБ. Позначка в Б чи КБ — інша
# помилка (кілька файлів одразу), її підказка лишається англійською.
# ponytail: селектори прив'язані до розмітки Streamlit 1.67 (data-testid); після оновлення
# Streamlit перевірити в браузері завантажувач (порожній, з файлом, завеликий файл) і числове
# поле. Межа: якщо перетягнути одразу кілька файлів у МБ, зайвий теж отримає підказку про розмір.
PLATFORM_CSS = """<style>
[data-testid="stFileUploaderDropzone"] button p {{display: none}}
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"]::after {{
  content: "Обрати файл"}}
[data-testid="stFileUploaderDropzone"] > div:not([data-testid]) > span,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="InputInstructions"] span {{visibility: hidden; font-size: 0}}
body:has([data-testid="stFileChip"][aria-invalid="true"][aria-label$="MB"])
  [data-testid="stTooltipErrorContent"] {{font-size: 0}}
body:has([data-testid="stFileChip"][aria-invalid="true"][aria-label$="MB"])
  [data-testid="stTooltipErrorContent"]::after {{
  content: "Файл має бути не більшим за {mb} МБ"; font-size: .875rem}}
[data-testid="stFileUploaderDropzone"] > div:not([data-testid]) > span::after {{
  content: "Перетягніть файл сюди"; visibility: visible; font-size: .875rem}}
[data-testid="stFileUploaderDropzoneInstructions"] span::after {{
  content: "Не більше {mb} МБ на файл"; visibility: visible; font-size: .875rem}}
[data-testid="InputInstructions"] span::after {{
  content: "Enter — застосувати"; visibility: visible; font-size: .75rem}}
</style>"""


def setup_page(title: str, icon: str) -> None:
    """Налаштування сторінки, єдиного стилю графіків і українських написів платформи."""
    st.set_page_config(page_title=f"{title} · Ризик ЦД2", page_icon=icon, layout="wide")
    st.html(PLATFORM_CSS.format(mb=st.get_option("server.maxUploadSize")))
    apply_style()


def page_header(title: str, what_happens: str) -> None:
    """Заголовок і блок «Що тут відбувається» простою мовою."""
    st.title(title)
    st.info(f"**Що тут відбувається.** {what_happens}")


@st.cache_data(show_spinner="Завантаження даних…")
def load_cohort() -> pd.DataFrame:
    """Аналітична вибірка NHANES з parquet."""
    return pd.read_parquet(PROCESSED_DIR / "cohort.parquet")


@st.cache_data(show_spinner="Завантаження даних…")
def load_flow() -> list[FlowStep]:
    """Кроки відбору учасників (для блок-схеми і таблиці)."""
    raw = json.loads((PROCESSED_DIR / "flow.json").read_text(encoding="utf-8"))
    return [FlowStep(**r) for r in raw]


@st.cache_data(show_spinner="Завантаження даних…")
def load_variables() -> pd.DataFrame:
    """Довідник змінних дослідження."""
    return pd.read_csv(PROCESSED_DIR / "variables.csv")


@st.cache_resource(show_spinner="Завантаження моделей…")
def get_models() -> tuple[dict, dict]:
    """Збережені моделі та метрики: завантажуються один раз за сесію сервера."""
    return load_models()


@st.cache_data(show_spinner="Завантаження моделей…")
def load_metrics() -> dict:
    """Вміст metrics.json збережених моделей."""
    return get_models()[1]


@st.cache_data(show_spinner="Завантаження даних…")
def load_predictions() -> pd.DataFrame:
    """Прогнози моделей і бали FINDRISC на тестовій вибірці."""
    return load_test_predictions()


def models_or_stop() -> tuple[dict, dict]:
    """Збережені моделі; якщо їх немає — повідомлення, підвал і зупинка сторінки."""
    try:
        return get_models()
    except ModelNotTrainedError:
        st.error(
            "Моделі ще не навчені. Щоб навчити їх, виконайте в теці проєкту "
            "`uv run python scripts/train_models.py` і оновіть сторінку."
        )
        disclaimer()
        st.stop()


def accept_user_file(content: bytes, name: str, file_id: str) -> list[str]:
    """Перевіряє файл: прийнятий замінює набір, відхилений очищає попередній."""
    user_df, errors = validate_user_file(content, name)
    if user_df is None:
        st.session_state.pop("user_df", None)
    else:
        st.session_state["user_df"] = user_df
    st.session_state["user_file_id"] = file_id
    st.session_state["user_errors"] = errors
    return errors


def active_dataset() -> tuple[pd.DataFrame, bool]:
    """Завантажений користувачем набір або аналітична вибірка NHANES; bool = чи користувацький."""
    user_df = st.session_state.get("user_df")
    if user_df is not None:
        return user_df, True
    return load_cohort(), False


def has_survey_design(df: pd.DataFrame) -> bool:
    """Чи є в наборі колонки ваг і дизайну (лише NHANES)."""
    return all(c in df.columns for c in SURVEY_COLS)


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Колонки, які є в наборі, мають дані і не є константою (для користувацьких файлів)."""
    return [c for c in cols if c in df.columns and df[c].nunique() > 1]


USER_STATE_KEYS = ("user_df", "user_file_id", "user_errors")


def uploader_key() -> str:
    """Ключ завантажувача файлів: змінюється при скиданні, щоб поле стало порожнім."""
    return f"user_upload_{st.session_state.get('uploader_counter', 0)}"


def on_upload_change() -> None:
    """Колбек завантажувача: забуває набір лише тоді, коли користувач сам прибрав файл."""
    if st.session_state.get(uploader_key()) is None:
        reset_user_data()


def reset_user_data() -> None:
    """Забуває файл користувача і очищає поле завантаження."""
    for key in USER_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["uploader_counter"] = st.session_state.get("uploader_counter", 0) + 1


def dataset_banner(is_user: bool, n: int, note: str = "") -> None:
    """Попередження, якщо завантажено файл користувача; note — що саме на сторінці не змінюється."""
    if not is_user:
        return
    st.warning(f"Розрахунки на завантаженому файлі (N = {fmt_int(n)}). {note}".strip())
    if st.button("Повернутися до NHANES", key="back_to_nhanes"):
        reset_user_data()
        st.rerun()


def nhanes_only_banner(note: str) -> None:
    """Банер для сторінок, що завжди показують NHANES, коли файл користувача завантажено."""
    user_df = st.session_state.get("user_df")
    dataset_banner(user_df is not None, 0 if user_df is None else len(user_df), note)


class Numbering:
    """Окремі лічильники рисунків і таблиць сторінки: номери 1.1, 1.2 і далі."""

    def __init__(self, page_no: int) -> None:
        self.page_no = page_no
        self._figures = 0
        self._tables = 0

    def figure(self) -> str:
        """Наступний номер рисунка."""
        self._figures += 1
        return f"{self.page_no}.{self._figures}"

    def table(self) -> str:
        """Наступний номер таблиці."""
        self._tables += 1
        return f"{self.page_no}.{self._tables}"


def _explain(how_to_read: str, meaning: str) -> None:
    left, right = st.columns(2)
    with left, st.expander("Як це читати", expanded=True):
        st.markdown(how_to_read)
    with right, st.expander("Що це означає", expanded=True):
        st.markdown(meaning)


def figure_block(fig: Figure, number: str, caption: str, how_to_read: str, meaning: str) -> None:
    """Рисунок, підпис, кнопка PNG і пояснення."""
    st.pyplot(fig)
    full = f"Рисунок {number}. {caption}"
    st.caption(full)
    png = fig_to_png(fig, 300)
    st.download_button(
        "Завантажити графік PNG (300 dpi)",
        data=png,
        file_name=f"рисунок_{number}.png",
        mime="image/png",
        key=f"dl_figure_{number}",
    )
    _explain(how_to_read, meaning)
    plt.close(fig)


def table_block(
    df: pd.DataFrame,
    number: str,
    caption: str,
    how_to_read: str,
    meaning: str,
    *,
    collapsed: bool = False,
    wrap: bool = False,
    shown: pd.DataFrame | None = None,
) -> None:
    """Таблиця, підпис і пояснення.

    collapsed=True ховає велику таблицю в блок, що розгортається, під своїм підписом.
    wrap=True — статична таблиця з перенесенням рядків: широкі таблиці й довгий текст
    вміщуються в ширину сторінки без горизонтального прокручування.
    shown — текстовий вигляд для екрана, якщо він відрізняється від df (Excel бере df).
    """
    view = df if shown is None else shown
    if collapsed:
        with st.expander(f"Таблиця {number}. {caption}"):
            _table_body(view, number, caption, how_to_read, meaning, wrap)
    else:
        _table_body(view, number, caption, how_to_read, meaning, wrap)


MAX_ROWS_SHOWN = 30  # таблиці до цієї довжини показуємо повністю, без внутрішнього прокручування
ROW_PX = 35  # висота рядка st.dataframe
NOT_COUNTS = frozenset({"SEQN", "Рік"})  # цілі колонки-коди й роки: без пробілу тисяч


def _decimals(column: pd.Series) -> int:
    """Скільки знаків після коми потрібно, щоб показати всі значення колонки точно."""
    texts = (f"{v:.10f}".rstrip("0") for v in column.dropna())
    return max((len(t.partition(".")[2]) for t in texts), default=0)


def _int_text(value: float) -> str:
    return fmt_int(round(value)).replace(" ", "\u00a0")  # число не рветься між рядками


def _float_text(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}".replace(",", "\u00a0")


def _text_cell(value: object) -> str:
    """Знак порівняння не відривається від числа: «< 0.001» не переноситься на два рядки."""
    return re.sub(r"([<>≤≥]) ", "\\1\u00a0", str(value))


def _count_columns(df: pd.DataFrame) -> list[str]:
    """Колонки-кількості: цілий тип або дроби без знаків після коми, крім кодів і років."""
    whole = [c for c in df.select_dtypes("float").columns if _decimals(df[c]) == 0]
    return [c for c in [*df.select_dtypes("integer").columns, *whole] if c not in NOT_COUNTS]


def _wrap_styler(df: pd.DataFrame) -> Styler:
    """Кількості — з пробілом тисяч; дроби — однакова кількість знаків у межах колонки."""
    formats: dict[str, Callable[[object], str] | Callable[[float], str]]
    formats = {c: _text_cell for c in df.columns}
    for col in df.select_dtypes("float").columns:
        formats[col] = partial(_float_text, digits=_decimals(df[col]))
    formats |= dict.fromkeys(_count_columns(df), _int_text)
    return df.style.format(formats, na_rep="—")


def _dataframe_data(df: pd.DataFrame) -> pd.DataFrame | Styler:
    """Дані для st.dataframe: кількості з пробілом тисяч, значення лишаються числами."""
    counts = _count_columns(df)
    return df.style.format(_int_text, subset=counts, na_rep="—") if counts else df


def _float_config(df: pd.DataFrame) -> dict:
    """Формат дробів у st.dataframe: однакова кількість знаків після коми в межах колонки."""
    counts = _count_columns(df)
    return {
        c: st.column_config.NumberColumn(format=f"%.{_decimals(df[c])}f")
        for c in df.select_dtypes("float").columns
        if c not in counts
    }


def _table_body(
    df: pd.DataFrame, number: str, caption: str, how_to_read: str, meaning: str, wrap: bool
) -> None:
    # службовий RangeIndex (0, 1, 2…) користувачу не потрібен; іменований індекс лишаємо
    hide_index = isinstance(df.index, pd.RangeIndex)
    if wrap:
        st.table(_wrap_styler(df), hide_index=hide_index)
    else:
        # до MAX_ROWS_SHOWN рядків — висота за кількістю рядків (заголовок + рядки + рамка)
        height = ROW_PX * (len(df) + 1) + 3 if len(df) <= MAX_ROWS_SHOWN else "auto"
        st.dataframe(
            _dataframe_data(df),
            width="stretch",
            hide_index=hide_index,
            height=height,
            column_config=_float_config(df),
        )
    st.caption(f"Таблиця {number}. {caption}")
    _explain(how_to_read, meaning)


def disclaimer() -> None:
    """Роздільник, DISCLAIMER і рядок автора дрібним шрифтом у кінці сторінки."""
    st.divider()
    st.caption(DISCLAIMER)
    st.caption(AUTHOR)


def fmt_int(n: int) -> str:
    """Ціле з пробілом тисяч: 11 304."""
    return f"{n:,}".replace(",", " ")


def group_name(value: float) -> str:
    """Назва групи за значенням t2d (0/1)."""
    return GROUP_LABELS[int(value)]


@st.cache_data(show_spinner="Обчислення…")
def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Кешована prevalence_all: рядок «Разом» + по циклах (англійські колонки)."""
    return prevalence_all(df)
