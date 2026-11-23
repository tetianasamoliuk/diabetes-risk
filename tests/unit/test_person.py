"""Тести risk.person."""

import math

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.pipeline import Pipeline

from risk.config import ACTIVE_MIN_PER_WEEK, FEATURES
from risk.errors import DataValidationError
from risk.findrisc.score import score_items
from risk.models.pipelines import make_hgb, make_logit
from risk.person import (
    BMI_SLIDER,
    CHANGE_COL,
    SCENARIO_PRESETS,
    PersonInput,
    applicable_presets,
    apply_scenario,
    bmi_from,
    chosen_scenarios,
    findrisc_result,
    format_scenarios,
    predict_all,
    readable_explanation,
    scenario_table,
    slider_bmi_start,
    to_feature_row,
    to_findrisc_row,
    train_medians,
    validate,
)


def make(**kw: float | bool | None) -> PersonInput:
    base = dict(
        age=40, male=True, height_cm=180, weight_kg=81, waist_cm=90, sbp=120, dbp=80,
        tc=190, hdl=50, pa_min_week=300, fam_dm=False, bp_med=False, high_glu_hist=False,
    )  # fmt: skip
    return PersonInput(**{**base, **kw})


def test_bmi() -> None:
    assert bmi_from(180, 81) == pytest.approx(25.0)
    with pytest.raises(DataValidationError):
        bmi_from(0, 70)


def test_valid() -> None:
    assert validate(make()) == []
    assert validate(make(sbp=None, dbp=None, tc=None, hdl=None)) == []


def test_all_errors_returned() -> None:
    p = make(age=10, height_cm=50, weight_kg=500, waist_cm=5, sbp=50, dbp=170,
             tc=1, hdl=300, pa_min_week=20000)  # fmt: skip
    errs = validate(p)
    assert "Вік має бути від 20 до 120 років" in errs
    assert "Діастолічний тиск має бути меншим за систолічний" in errs
    assert any("ІМТ" in e for e in errs)
    assert len(errs) >= 10


def test_dbp_not_less_than_sbp() -> None:
    assert validate(make(sbp=100, dbp=100)) == ["Діастолічний тиск має бути меншим за систолічний"]


def test_bmi_out_of_range_message() -> None:
    # 30 / 2.5² = 4.8: зріст і вага в межах, але ІМТ поза діапазоном
    assert validate(make(height_cm=250, weight_kg=30)) == [
        "Отриманий ІМТ 4.8 поза правдоподібним діапазоном — перевірте зріст і вагу"
    ]
    assert validate(make(height_cm=100, weight_kg=300)) == [
        "Отриманий ІМТ 300.0 поза правдоподібним діапазоном — перевірте зріст і вагу"
    ]


def test_zero_height_no_crash() -> None:
    assert any("Зріст" in e for e in validate(make(height_cm=0)))


def test_negative_height_no_crash() -> None:
    assert validate(make(height_cm=-170)) == ["Зріст має бути від 100 до 250 см"]


def test_range_message_has_thousands_separator() -> None:
    assert validate(make(pa_min_week=20000)) == [
        "Фізична активність має бути від 0 до 10 080 хв/тиждень"
    ]


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("age", "Вік"),
        ("height_cm", "Зріст"),
        ("weight_kg", "Вага"),
        ("waist_cm", "Окружність талії"),
        ("pa_min_week", "Фізична активність"),
    ],
)
def test_empty_required_field_is_named(field: str, label: str) -> None:
    # порожній зріст чи вага не ламають розрахунок ІМТ
    assert validate(make(**{field: None})) == [f"{label}: введіть значення"]


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("sbp", "Систолічний тиск"),
        ("dbp", "Діастолічний тиск"),
        ("tc", "Загальний холестерин"),
        ("hdl", "ЛПВЩ-холестерин"),
    ],
)
def test_empty_optional_field_without_unknown(field: str, label: str) -> None:
    assert validate(make(**{field: None}), empty=[field]) == [
        f"{label}: введіть значення або позначте «не знаю»"
    ]
    assert validate(make(**{field: None})) == []  # None без empty — «не знаю»


def test_fractional_weight_accepted() -> None:
    assert validate(make(weight_kg=75.5)) == []


def test_feature_row() -> None:
    df = to_feature_row(make(sbp=None, tc=None))
    assert list(df.columns) == FEATURES
    assert len(df) == 1
    r = df.iloc[0]
    assert r["male"] == 1.0 and r["pa_h_week"] == 5.0 and r["cycle_2017"] == 1.0
    assert r["bmi"] == pytest.approx(25.0)
    assert math.isnan(r["sbp"]) and math.isnan(r["tc"]) and r["dbp"] == 80
    assert to_feature_row(make(), cycle_2017=0.0).iloc[0]["cycle_2017"] == 0.0


def test_findrisc_max_example() -> None:
    p = make(age=70, height_cm=180, weight_kg=110, waist_cm=103, pa_min_week=0,
             bp_med=True, high_glu_hist=True, fam_dm=True)  # fmt: skip
    items = score_items(to_findrisc_row(p))
    assert items.iloc[0].tolist() == [4, 3, 4, 2, 2, 5, 5]
    assert items.iloc[0].sum() == 25
    assert findrisc_result(p) == (25, "дуже високий")


def test_findrisc_zero_example() -> None:
    p = make(age=30, male=False, height_cm=170, waist_cm=75, pa_min_week=300)
    p = apply_scenario(p, bmi_target=22)
    assert score_items(to_findrisc_row(p)).iloc[0].sum() == 0


def test_scenario_bmi_changes_weight_not_height() -> None:
    p = make()
    q = apply_scenario(p, bmi_target=22)
    assert q.height_cm == p.height_cm and q.weight_kg != p.weight_kg
    assert bmi_from(q.height_cm, q.weight_kg) == pytest.approx(22)


def test_scenario_active_and_waist() -> None:
    p = make(pa_min_week=0)
    assert apply_scenario(p, active=True).pa_min_week >= ACTIVE_MIN_PER_WEEK
    assert apply_scenario(make(pa_min_week=500), active=True).pa_min_week == 500
    assert apply_scenario(make(), active=False).pa_min_week == 0
    assert apply_scenario(p, waist_delta=-10).waist_cm == 80


def test_scenario_unknown_field() -> None:
    with pytest.raises(TypeError):
        apply_scenario(make(), foo=1)


def test_presets_apply() -> None:
    assert 3 <= len(SCENARIO_PRESETS) <= 4
    p = make(bp_med=True, weight_kg=95, pa_min_week=0)
    by_name = {s["name"]: apply_scenario(p, **s["changes"]) for s in SCENARIO_PRESETS}
    assert set(by_name) == {
        "Схуднути до ІМТ 25",
        "Талія −10 см",
        f"Стати активним ≥ {ACTIVE_MIN_PER_WEEK // 7} хв/день",
    }
    bmi, waist, active = by_name.values()
    assert bmi_from(bmi.height_cm, bmi.weight_kg) == pytest.approx(25) and bmi.waist_cm == 90
    assert waist.waist_cm == 80 and waist.weight_kg == 95
    assert active.pa_min_week == ACTIVE_MIN_PER_WEEK and active.waist_cm == 90


@pytest.fixture(scope="module")
def models() -> dict[str, Pipeline]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    y = (X["bmi"] + rng.normal(size=200) > 0).astype(int)
    return {"logit": make_logit().fit(X, y), "hgb": make_hgb().fit(X, y)}


def test_predict_all(models: dict[str, Pipeline]) -> None:
    probs = predict_all(models, make(sbp=None))
    assert set(probs) == {"logit", "hgb"}
    assert all(0 <= v <= 1 for v in probs.values())


def test_high_glu_hist_not_in_model_inputs(models: dict[str, Pipeline]) -> None:
    a, b = make(high_glu_hist=False), make(high_glu_hist=True)
    pd.testing.assert_frame_equal(to_feature_row(a), to_feature_row(b))
    assert predict_all(models, a) == predict_all(models, b)


def test_findrisc_result() -> None:
    # вік 40→0, ІМТ 25→1, талія 90 (чол.)→0, активний→0, без ліків→0
    assert findrisc_result(make()) == (1, "низький")
    # +5 глюкоза, +5 спадковість, +2 ліки: 1+12 = 13
    assert findrisc_result(make(high_glu_hist=True, fam_dm=True, bp_med=True)) == (13, "помірний")


def test_scenario_table(models: dict[str, Pipeline]) -> None:
    t = scenario_table(models, make(), SCENARIO_PRESETS, best="hgb")
    assert len(t) == 1 + len(SCENARIO_PRESETS)
    assert t["Сценарій"].iloc[0] == "Поточний стан"
    assert t["Зміна (основна модель), п.п."].iloc[0] == 0
    cur = predict_all(models, make())["hgb"] * 100
    assert t["hgb"].iloc[0] == pytest.approx(cur)
    assert (t.iloc[1:]["hgb"] - cur).tolist() == pytest.approx(
        t.iloc[1:]["Зміна (основна модель), п.п."].tolist()
    )


def test_train_medians_and_hgb_equivalence(models: dict[str, Pipeline]) -> None:
    med = train_medians(models)
    assert list(med) == FEATURES
    p = make(sbp=None, dbp=None, tc=None, hdl=None)
    filled = make(sbp=med["sbp"], dbp=med["dbp"], tc=med["tc"], hdl=med["hdl"])
    got = predict_all(models, p, medians=med)
    assert got == pytest.approx(predict_all(models, filled))
    # з medians пропуск у рядку ознак заповнено
    assert not math.isnan(to_feature_row(p, medians=med).iloc[0]["tc"])


def test_train_medians_without_imputer() -> None:
    with pytest.raises(DataValidationError):
        train_medians({"hgb": make_hgb()})


def test_applicable_presets() -> None:
    heavy = [s["name"] for s in applicable_presets(make(weight_kg=95))]
    assert "Схуднути до ІМТ 25" in heavy
    for w in (81, 60):  # ІМТ = 25 і < 25
        names = [s["name"] for s in applicable_presets(make(weight_kg=w))]
        assert "Схуднути до ІМТ 25" not in names
        assert len(names) == len(SCENARIO_PRESETS) - 1


def test_scenario_table_skips_out_of_range(models: dict[str, Pipeline]) -> None:
    scenarios = [
        {"name": "ok", "changes": {"waist_delta": -10}},
        {"name": "ok2", "changes": {"waist_delta": -5}},
    ]
    p = make(waist_cm=45)  # 35 см < мінімуму
    t = scenario_table(models, p, scenarios, best="hgb")
    assert t["Сценарій"].tolist() == ["Поточний стан", "ok2"]  # "ok" -> 35 см пропущено
    assert t.attrs["skipped"] == ["ok"]
    t = scenario_table(models, make(waist_cm=90), scenarios, best="hgb")
    assert t["Сценарій"].tolist() == ["Поточний стан", "ok", "ok2"]
    assert t.attrs["skipped"] == []


def test_train_medians_feature_order_mismatch() -> None:
    class Imp:
        feature_names_in_ = np.array(FEATURES[::-1])
        statistics_ = np.zeros(len(FEATURES))

    fake = Pipeline([("impute", Imp())])
    with pytest.raises(DataValidationError, match="Порядок колонок"):
        train_medians({"logit": fake})


def test_total_cholesterol_below_hdl_is_error() -> None:
    assert validate(make(tc=100, hdl=200)) == [
        "Загальний холестерин не може бути меншим за ЛПВЩ-холестерин"
    ]
    assert validate(make(tc=None, hdl=200)) == []


def test_slider_bmi_start_is_clamped_to_slider() -> None:
    assert slider_bmi_start(make()) == 25.0
    assert slider_bmi_start(make(weight_kg=300, height_cm=150)) == BMI_SLIDER[1]
    assert slider_bmi_start(make(weight_kg=40, height_cm=190)) == BMI_SLIDER[0]


def test_chosen_scenarios_presets_and_custom() -> None:
    p = make(weight_kg=95)
    start = slider_bmi_start(p)
    names = [s["name"] for s in applicable_presets(p)]
    only_presets = chosen_scenarios(p, names, bmi_target=start)
    assert [s["name"] for s in only_presets] == names
    assert chosen_scenarios(p, [], bmi_target=start + 0.04) == []
    custom = chosen_scenarios(p, names[:1], bmi_target=start - 2, waist_delta=-5, active=True)
    assert [s["name"] for s in custom] == [names[0], "Мій сценарій"]
    assert custom[-1]["changes"] == {"bmi_target": start - 2, "waist_delta": -5, "active": True}
    unknown = chosen_scenarios(p, ["немає такого"], bmi_target=start)
    assert unknown == []


def test_format_scenarios_has_constant_format(models: dict[str, Pipeline]) -> None:
    t = scenario_table(models, make(), SCENARIO_PRESETS, best="hgb")
    shown = format_scenarios(t, list(models))
    cells = shown[list(models)].to_numpy().ravel()
    assert all(c.endswith(" %") and c[:-2].replace(".", "").isdigit() for c in cells)
    assert shown[CHANGE_COL].iloc[0] == "0.0 п.п."
    assert all(c.endswith(" п.п.") for c in shown[CHANGE_COL])


def test_readable_explanation_labels_and_rounding() -> None:
    names = ["bmi", "fam_dm", "cycle_2017", "waist"]
    expl = shap.Explanation(
        values=np.array([0.1, -0.5, 0.2, 0.0]),
        base_values=-2.0,
        data=np.array([25.952, 0.0, 1.0, 90.0]),
        feature_names=names,
    )
    out = readable_explanation(expl)
    assert list(out.data) == ["26.0", "ні", "так", "90"]
    assert list(out.feature_names)[2] == "Період обстеження: 2017–2020"
    assert list(out.values) == [0.1, -0.5, 0.2, 0.0] and out.base_values == -2.0
