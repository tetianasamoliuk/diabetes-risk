"""Тести автопояснень: межі шкал, форматування чисел, відсутність захардкожених результатів."""

import pytest

from risk import interpret as it
from risk.config import ALPHA
from risk.stats.hypothesis import MIN_NORMALITY_N


def test_fmt_p() -> None:
    assert it.fmt_p(0.0004) == "< 0.001"
    assert it.fmt_p(0.001) == "0.001"
    assert it.fmt_p(0.049) == "0.0490"
    assert it.fmt_p(0.0499) == "0.0499"
    assert it.fmt_p(0.05) == "0.0500"
    assert it.fmt_p(0.0549) == "0.0549"
    assert it.fmt_p(0.055) == "0.055"
    assert it.fmt_p(0.044) == "0.044"
    assert it.fmt_num(2.4567) == "2.46"
    assert it.fmt_num(2.4567, 1) == "2.5"


@pytest.mark.parametrize(("p", "sig"), [(0.049, True), (0.05, False), (0.051, False)])
def test_p_value_boundaries(p: float, sig: bool) -> None:
    text = it.p_value_text(p)
    assert it.fmt_p(p) in text
    assert ("не є статистично значущою" in text) is not sig
    assert ("не відхиляємо" in text) is not sig


def test_p_value_tiny_and_alpha() -> None:
    assert "< 0.001" in it.p_value_text(1e-9)
    assert "не є" in it.p_value_text(0.03, alpha=0.01)


@pytest.mark.parametrize(
    ("lo", "hi", "contains"),
    [(0.9, 1.1, True), (1.1, 1.3, False), (1.0, 1.4, True), (0.6, 1.0, True), (0.5, 0.9, False)],
)
def test_or_ci_one(lo: float, hi: float, contains: bool) -> None:
    text = it.or_text("ожирінням", 1.2, lo, hi)
    assert ("Інтервал містить 1" in text) is contains
    assert ("Інтервал не містить 1" in text) is not contains
    assert f"{lo:.2f}–{hi:.2f}" in text


def test_or_text_continuous_scale() -> None:
    text = it.or_text("Вік", 1.62, 1.5, 1.75, scale_label="10 років")
    assert "Кожні 10 років" in text and "у 1.62 раза" in text and "1.50–1.75" in text
    assert "Інтервал не містить 1" in text


def test_or_directions_and_reference() -> None:
    up = it.or_text("ІМТ > 30", 2.4, 2.1, 2.8, "у людей з нормальною вагою")
    assert "2.40 раза вищі" in up and "ніж у людей з нормальною вагою" in up
    assert "ніж у решти" in it.or_text("X", 2.0, 1.5, 2.5)
    down = it.or_text("X", 0.5, 0.4, 0.6)
    assert "становлять 0.50 від шансів у решти" in down
    same = it.or_text("X", 1.0, 0.9, 1.1)
    assert "не відрізняються" in same


@pytest.mark.parametrize(
    ("auc", "word"),
    [
        (0.5, "слабка"), (0.65, "слабка"), (0.699, "слабка"),
        (0.7, "прийнятна"), (0.799, "прийнятна"), (0.8, "відмінна"),
        (0.899, "відмінна"), (0.9, "видатна"), (0.97, "видатна"),
    ],
)  # fmt: skip
def test_auc_scale(auc: float, word: str) -> None:
    text = it.auc_text(auc)
    assert word in text and f"{auc:.2f}" in text
    assert "випадковий хворий" in text and "ДІ" not in text


def test_auc_ci() -> None:
    text = it.auc_text(0.81, 0.77, 0.85)
    assert "95% ДІ 0.77–0.85" in text


def test_test_choice_note_uses_config_values() -> None:
    note = it.TEST_CHOICE_NOTE
    assert "нормальн" in note and str(ALPHA) in note and str(MIN_NORMALITY_N) in note
    assert "Велча" in note and "Манна–Вітні" in note and "χ²" in note


def test_prevalence_text_uses_ukrainian_interval() -> None:
    text = it.prevalence_text(12.3, 11.0, 13.6, 15.1)
    assert "95% ДІ" in text and "CI" not in text


def test_calibration() -> None:
    good = it.calibration_text(0.12, 0.151, 0.15, 1.03)
    assert "Калібрування добре" in good and "0.12" in good and "1.03" in good
    over = it.calibration_text(0.2, 0.25, 0.15, 1.0)
    assert "завищує" in over and "10.0 п.п." in over and "Калібрування добре" not in over
    under = it.calibration_text(0.2, 0.10, 0.15, 1.0)
    assert "занижує" in under and "5.0 п.п." in under
    assert "завищує" in it.calibration_text(0.1, 0.17, 0.15, 1.0)  # рівно 2 п.п. — вже не «добре»
    assert "Калібрування добре" in it.calibration_text(0.1, 0.169, 0.15, 1.0)


def test_calibration_slope_overrides_mean() -> None:
    # середні збігаються, але нахил помітно відхиляється від 1 — «добре» писати не можна
    steep = it.calibration_text(0.12, 0.151, 0.15, 0.84)
    assert "0.84" in steep and "менший за 1" in steep and "надто крайні" in steep
    flat = it.calibration_text(0.12, 0.151, 0.15, 1.23)
    assert "1.23" in flat and "більший за 1" in flat and "стиснуті" in flat
    for text in (steep, flat):
        assert "Калібрування добре" not in text and "орієнтир" in text
    assert "Калібрування добре" in it.calibration_text(0.12, 0.151, 0.15, 0.92)


def test_delong_first_is_chosen_model() -> None:
    a = it.delong_text("Random Forest", 0.82, "Gradient Boosting", 0.83, 0.372)
    assert a.startswith("Random Forest: AUC 0.820; Gradient Boosting: AUC 0.830.")
    assert "Різниця −0.010 не є статистично значущою (p = 0.372)" in a
    assert "Найкраща" not in a  # вибір моделі — за крос-валідацією, не за цим текстом
    s = it.delong_text("A", 0.8, "B", 0.7, 1e-5)
    assert "Різниця +0.100 є статистично значущою (p < 0.001)" in s


def test_delong_text_with_holm() -> None:
    lost = it.delong_text("A", 0.825, "B", 0.814, 0.024, p_holm=0.072)
    assert "Різниця +0.011 не є статистично значущою з поправкою Холма" in lost
    assert "сире p = 0.024, з поправкою p = 0.072" in lost
    assert "Без поправки вона виглядала б значущою." in lost
    kept = it.delong_text("A", 0.8, "B", 0.7, 1e-9, p_holm=2e-9)
    assert "є статистично значущою з поправкою Холма" in kept and "Без поправки" not in kept
    none = it.delong_text("A", 0.8, "B", 0.79, 0.7, p_holm=0.7)
    assert "не є статистично значущою" in none and "Без поправки" not in none


def test_shap() -> None:
    labels = {"age": "Вік", "bmi": "ІМТ", "hdl": "ЛПВЩ"}
    text = it.shap_text([("age", 0.52), ("hdl", -0.3), ("bmi", 0.1)], labels)
    assert "Вік (+0.52)" in text and "ІМТ (+0.10)" in text and "ЛПВЩ (−0.30)" in text
    assert text.index("Вік") < text.index("ІМТ")
    assert "log-odds" in text
    assert "log-odds (логарифм шансів)" in it.SHAP_AXES_NOTE
    only_pos = it.shap_text([("x", 0.2)], {})
    assert "x (+0.20)" in only_pos and "знизили" not in only_pos
    assert "Найбільше підвищили" not in it.shap_text([("x", -0.2)], {})
    assert it.shap_text([], labels)


def test_prevalence() -> None:
    text = it.prevalence_text(12.3, 11.0, 13.6, 15.1)
    for s in ("12.3%", "11.0–13.6%", "15.1%"):
        assert s in text
    assert "ваг опитування" in text


@pytest.mark.parametrize(
    ("name", "ukr"),
    [("Welch", "t-критерій Велча"), ("mann-whitney", "критерій Манна–Вітні"),
     ("chi2", "критерій χ²"), ("інший", "інший")],
)  # fmt: skip
def test_test_text(name: str, ukr: str) -> None:
    text = it.test_text("Вік", name, 3.456, 0.0123, 0.049)
    assert ukr in text and "3.46" in text
    assert "сире p = 0.012" in text and "після поправки Холма p = 0.049" in text
    assert "є статистично значущою" not in it.test_text("Вік", name, 1, 0.01, 0.06).replace(
        "не є статистично значущою", ""
    )


def test_threshold() -> None:
    text = it.threshold_text(0.23, 0.712, 0.655)
    for s in ("23.0 %", "71.2%", "65.5%", "Юдена"):
        assert s in text
    assert "хворих менше" in text


def test_no_hardcoded_results() -> None:
    pairs = [
        (it.p_value_text(0.01), it.p_value_text(0.4)),
        (it.or_text("a", 2, 1.5, 3), it.or_text("a", 3, 2.5, 4)),
        (it.auc_text(0.65), it.auc_text(0.85)),
        (it.calibration_text(0.1, 0.2, 0.1, 1.0), it.calibration_text(0.3, 0.1, 0.2, 0.8)),
        (it.delong_text("A", 0.7, "B", 0.8, 0.01), it.delong_text("A", 0.71, "B", 0.75, 0.3)),
        (it.shap_text([("a", 1.0)], {}), it.shap_text([("a", 2.0)], {})),
        (it.prevalence_text(10, 9, 11, 20), it.prevalence_text(20, 19, 21, 30)),
        (it.test_text("v", "welch", 1, 0.01, 0.02), it.test_text("v", "welch", 2, 0.5, 0.6)),
        (it.threshold_text(0.2, 0.7, 0.6), it.threshold_text(0.3, 0.8, 0.5)),
    ]
    for x, y in pairs:
        assert x and y and x != y


def test_vif_text() -> None:
    import pandas as pd

    high = pd.DataFrame({"vif": [8.2, 7.9, 1.2]}, index=["bmi", "waist", "age"])
    text = it.vif_text(high)
    assert "«Індекс маси тіла»" in text and "«Окружність талії»" in text
    assert "«Вік»" not in text
    assert "8.2" in text and "обережно" in text
    assert "колінеарність незначна" in it.vif_text(
        pd.DataFrame({"vif": [1.2, 2.0]}, index=["a", "b"])
    )


def test_shap_text_uses_values_and_neutral_wording() -> None:
    labels = {"fam_dm": "Діабет у близьких родичів", "tc": "Холестерин"}
    contrib = [("fam_dm", -0.54), ("waist", -0.36), ("bmi", 0.3), ("tc", 0.1)]
    values = {"fam_dm": 0, "waist": 90, "bmi": 27.14, "tc": float("nan")}
    text = it.shap_text(contrib, labels, values)
    assert text.startswith("Найбільше підвищили оцінку моделі: ІМТ 27.1 кг/м² (+0.30), ")
    assert "Холестерин (+0.10)" in text
    assert (
        "Найбільше знизили: відсутність діабету в близьких родичів (−0.54), "
        "окружність талії 90 см (−0.36)" in text
    )
    assert "немає:" not in text and "підвищили ризик" not in text


def test_shap_text_without_any_direction() -> None:
    assert "Жоден чинник" in it.shap_text([("age", 0.0)], {})


def test_feature_phrase_binary_and_unknown() -> None:
    assert it.feature_phrase("male", 1) == "чоловіча стать"
    assert it.feature_phrase("cycle_2017", 0) == "обстеження в періоді 2015–2016"
    assert it.feature_phrase("pa_h_week", 2.54) == "фізична активність 2.5 год/тиждень"
    assert it.feature_phrase("zzz", 1) is None and it.feature_phrase("age", None) is None


def test_feature_display_for_waterfall() -> None:
    assert it.feature_display("fam_dm", 0.0) == ("ні", "Діабет у близьких родичів")
    assert it.feature_display("cycle_2017", 1.0) == ("так", "Період обстеження: 2017–2020")
    assert it.feature_display("bmi", 25.952) == ("26.0", "Індекс маси тіла, кг/м²")
    assert it.feature_display("waist", 90.0) == ("90", "Окружність талії, см")
    assert it.feature_display("zzz", 3.0) == ("3", "zzz")


def test_auc_text_conclusion_has_no_noun_model() -> None:
    assert "Висновок: розрізнювальна здатність — прийнятна." in it.auc_text(0.75)


@pytest.mark.parametrize(
    ("n", "word"),
    [(1, "помилка"), (21, "помилка"), (2, "помилки"), (24, "помилки"), (5, "помилок"),
     (11, "помилок"), (12, "помилок"), (20, "помилок"), (111, "помилок")],
)  # fmt: skip
def test_plural_uk(n: int, word: str) -> None:
    assert it.plural_uk(n, ("помилка", "помилки", "помилок")) == word


def test_scenario_formats() -> None:
    assert it.fmt_pct(2.04) == "2.0 %"
    assert it.fmt_pp(-0.2) == "−0.2 п.п." and it.fmt_pp(0.34) == "+0.3 п.п."
    assert it.fmt_pp(-0.04) == "0.0 п.п." and it.fmt_pp(0) == "0.0 п.п."


def test_fmt_cell() -> None:
    assert it.fmt_cell(float("nan")) == "—" and it.fmt_cell(float("inf")) == "—"
    assert it.fmt_cell(None) == "—"
    assert it.fmt_cell(1.0008, 2) == "1.00"
    assert it.fmt_cell(1e15, cap=1000) == "> 1000" and it.fmt_cell(999.0, cap=1000) == "999.00"


def test_logit_formula_is_split_into_lines_with_ukrainian_names() -> None:
    coefs = {"const": -5.984, **{n: 0.01 * (i + 1) for i, n in enumerate(it.FORMULA_NAMES_UK)}}
    coefs["bmi"] = -0.02
    latex = it.logit_formula_aligned(coefs)
    assert latex.count("\\\\") == 3 and "\\hat p" in latex and "z &= -5.984 +" in latex
    assert "- 0.020\\,\\text{ІМТ}" in latex and "age" not in latex and "cycle" not in latex
    assert "\\text{період 2017–2020}" in latex


def test_logit_legend_mentions_every_formula_name() -> None:
    for name in it.FORMULA_NAMES_UK.values():
        assert name in it.LOGIT_LEGEND
    assert "p̂" in it.LOGIT_LEGEND and "β" in it.LOGIT_LEGEND
