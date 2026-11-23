"""Сторінка «Оцінка ризику» через AppTest: форма, перевірки, SHAP, сценарії."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[2] / "app" / "pages" / "7_👤_Оцінка_ризику.py"


def run() -> AppTest:
    return AppTest.from_file(str(PAGE), default_timeout=240).run()


def _texts(at: AppTest) -> str:
    return " ".join(e.value for e in [*at.markdown, *at.warning, *at.info, *at.caption])


def test_opens_without_exception() -> None:
    at = run()
    assert not at.exception
    assert len(at.metric) == 0
    assert any("не медичним діагнозом" in w.value for w in at.warning)


def test_submit_valid() -> None:
    at = run()
    at.button[0].click().run()
    assert not at.exception
    assert "FINDRISC" in _texts(at)
    assert len(at.metric) >= 3
    assert len(at.dataframe) >= 1


def test_default_values_on_first_open() -> None:
    at = run()
    assert [n.value for n in at.number_input] == [45, 170, 75, 90, 150, 120, 80, 190, 50]


def test_age_out_of_range_shows_message_not_result() -> None:
    at = run()
    at.number_input[0].set_value(5)
    at.button[0].click().run()
    assert [e.value for e in at.error] == ["Вік має бути від 20 до 120 років"]
    assert len(at.metric) == 0


def test_empty_height_shows_message_not_result() -> None:
    at = run()
    at.number_input[1].set_value(None)
    at.button[0].click().run()
    assert not at.exception
    assert [e.value for e in at.error] == ["Зріст: введіть значення"]
    assert len(at.metric) == 0


def test_empty_optional_field_without_unknown_shows_message() -> None:
    at = run()
    at.number_input[5].set_value(None)  # САТ порожній, «не знаю» не відмічено
    at.button[0].click().run()
    assert [e.value for e in at.error] == [
        "Систолічний тиск: введіть значення або позначте «не знаю»"
    ]
    assert len(at.metric) == 0


def test_unknown_disables_field() -> None:
    at = run()
    assert not at.number_input(key="sbp").disabled
    at.checkbox(key="sbp_unknown").check().run()
    assert at.number_input(key="sbp").disabled
    at.button[0].click().run()
    assert not at.error and len(at.metric) >= 3


def test_fractional_weight_kept() -> None:
    at = run()
    at.number_input[2].set_value(75.5)
    at.button[0].click().run()
    assert at.session_state["person"].weight_kg == 75.5


def test_invalid_bmi() -> None:
    # зріст і вага в межах, але ІМТ 30 / 2.5² = 4.8 поза діапазоном
    at = run()
    at.number_input[1].set_value(250)
    at.number_input[2].set_value(30)
    at.button[0].click().run()
    assert any("ІМТ" in e.value and "перевірте" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_dbp_not_below_sbp() -> None:
    at = run()
    at.number_input[6].set_value(130)  # ДАТ > САТ (120)
    at.button[0].click().run()
    assert any("Діастолічний" in e.value for e in at.error)


def _first_table(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if "Сценарій" in d.value.columns)


def test_out_of_range_value_shows_error_not_previous_result() -> None:
    at = run()
    at.button[0].click().run()
    assert len(at.metric) >= 3
    at.number_input[1].set_value(250)  # зріст 250 + вага 30 → ІМТ поза діапазоном
    at.number_input[2].set_value(30)
    at.run()
    assert at.error and len(at.metric) == 0


def test_total_cholesterol_below_hdl_is_error() -> None:
    at = run()
    at.number_input[7].set_value(100)  # загальний холестерин
    at.number_input[8].set_value(200)  # ЛПВЩ
    at.button[0].click().run()
    assert any("ЛПВЩ" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_changed_input_hides_old_result_and_warns() -> None:
    at = run()
    at.button[0].click().run()
    at.number_input[0].set_value(60)
    at.run()
    assert any("Дані змінено — натисніть «Оцінити ризик»" in w.value for w in at.warning)
    assert len(at.metric) == 0
    at.button[0].click().run()
    assert len(at.metric) >= 3 and not any("Дані змінено" in w.value for w in at.warning)


def test_improving_scenarios_do_not_raise_risk() -> None:
    """Монотонні обмеження: схуднення, менша талія, активність не підвищують оцінку."""
    at = run()
    at.button[0].click().run()
    table = _first_table(at)
    assert any("Схуднути" in s for s in table["Сценарій"])
    assert not any(v.startswith("+") for v in table["Зміна (основна модель), п.п."])


def test_scenario_table_has_constant_format() -> None:
    at = run()
    at.button[0].click().run()
    table = _first_table(at)
    cells = table.drop(columns=["Сценарій", "Зміна (основна модель), п.п."]).to_numpy().ravel()
    assert all(str(c).endswith(" %") and "." in str(c) for c in cells)
    assert all(str(c).endswith(" п.п.") for c in table["Зміна (основна модель), п.п."])


def test_shap_caption_and_text_for_gradient_boosting() -> None:
    at = run()
    at.button[0].click().run()
    captions = [c.value for c in at.caption]
    assert any(
        c.startswith("Рисунок 7.1. Внесок ознак у прогноз моделі Gradient Boosting (її ймовірність")
        and c.endswith("%)")
        for c in captions
    )
    text = _texts(at)
    assert "E[f(X)] — середній прогноз на навчальній вибірці" in text
    assert "Найбільше знизили" in text and "немає:" not in text
    assert "підвищили ризик" not in text


def test_slider_bmi_has_one_decimal_format() -> None:
    at = run()
    at.button[0].click().run()
    bmi = next(s for s in at.slider if s.label == "Цільовий ІМТ")
    assert bmi.proto.format == "%.1f"


def test_cycle_and_nonmonotone_notes_shown() -> None:
    at = run()
    at.button[0].click().run()
    captions = [c.value for c in at.caption]
    assert any(c.startswith("Рядок «Період обстеження» — не ваш показник") for c in captions)
    assert any("нижчу оцінку при вищому значенні" in c for c in captions)
