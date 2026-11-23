"""Smoke-тести сторінок «Моделі» і «FINDRISC»."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from risk.config import MODELS_DIR
from risk.data.user_data import validate_user_file
from tests.app.conftest import user_csv

PAGES = Path(__file__).resolve().parents[2] / "app" / "pages"
MODELS = PAGES / "4_🧮_Моделі.py"
FINDRISC = PAGES / "5_📋_FINDRISC.py"


def run(path: Path) -> AppTest:
    return AppTest.from_file(str(path), default_timeout=240).run()


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_models_page() -> None:
    at = run(MODELS)
    assert not at.exception
    assert len(at.get("latex")) >= 1
    assert len(at.dataframe) + len(at.table) >= 4  # 4.1–4.4
    # широка таблиця метрик — статична з перенесенням, усі колонки видно
    metrics_table = next(t.value for t in at.table if "Специфічність" in t.value.columns)
    assert "Чутливість (recall)" in metrics_table.columns
    # висновок про моделі — з даних, з поправкою Холма, без зеленого «успіху»
    assert not at.success
    assert any("поправкою Холма" in i.value for i in at.info)
    assert "Рисунок 4.7. SHAP: середня важливість ознак у прогнозі Gradient Boosting" in _captions(
        at
    )
    texts = " ".join(m.value for m in at.markdown)
    assert "Нахил калібрування" in texts
    # підписи називають вибірку, англійських назв метрик немає
    assert "тестова вибірка.**" in texts
    assert "тест ДеЛонга, тестова вибірка)" in " ".join(_captions(at))
    for word in ("CV AUC", "Test AUC", "Recall", "Specificity", "train", "Brier"):
        assert word not in texts
    assert "стандартне відхилення" in texts


def test_findrisc_page() -> None:
    at = run(FINDRISC)
    assert not at.exception
    assert len(at.dataframe) >= 3
    captions = _captions(at)
    order = [
        "Таблиця 5.3. AUC трьох варіантів FINDRISC на тестовій вибірці",
        "Таблиця 5.5. AUC FINDRISC на всій аналітичній вибірці (описово)",
    ]
    assert all(c in captions for c in order)
    assert any(c.startswith("Таблиця 5.6. AUC моделей і FINDRISC (a) окремо") for c in captions)
    texts = " ".join(m.value for m in at.markdown)
    # головна оцінка шкали — на всій вибірці, і вона йде першою
    assert texts.index("Уся аналітична вибірка, особи з повною анкетою") < texts.index(
        "тестова вибірка."
    )
    assert "особи з повною анкетою, варіант (a)" in texts
    assert "модель бачить виміряні тиск" in texts
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    for n in (metrics["n_test"], metrics["findrisc"]["b"]["n"]):
        assert f"{n:,}".replace(",", " ") in texts + " ".join(_captions(at))
    assert "для виявлення недіагностованого діабету вона не доведена" in texts
    assert len(at.table) >= 2  # 5.1 і 5.6 — з перенесенням рядків


def test_findrisc_page_on_full_user_data() -> None:
    rng = np.random.default_rng(1)
    n = 200
    raw = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n), "male": rng.integers(0, 2, n),
            "bmi": rng.normal(28, 4, n).clip(15, 60), "waist": rng.normal(95, 10, n).clip(60, 150),
            "pa_h_week": rng.uniform(0, 10, n).round(1), "bp_med": rng.integers(0, 2, n),
            "fam_dm": rng.integers(0, 2, n), "t2d": rng.integers(0, 2, n),
        }
    )  # fmt: skip
    user_df, errors = validate_user_file(raw.to_csv(index=False).encode(), "x.csv")
    assert user_df is not None, errors
    at = AppTest.from_file(str(FINDRISC), default_timeout=240)
    at.session_state["user_df"] = user_df
    at.run()
    assert not at.exception
    assert any("варіант (a)" in i.value for i in at.info)
    assert not any("FINDRISC для вашого файлу" in w.value for w in at.warning)


def test_findrisc_page_on_user_data() -> None:
    user_df, errors = validate_user_file(user_csv(), "x.csv")
    assert user_df is not None, errors
    at = AppTest.from_file(str(FINDRISC), default_timeout=240)
    at.session_state["user_df"] = user_df
    at.run()
    assert not at.exception
    # без колонок активності/тиску/спадковості FINDRISC не рахується — сторінка каже це явно
    warning = next(w.value for w in at.warning if "FINDRISC" in w.value)
    assert "pa_h_week — Фізична активність" in warning and "fam_dm — Діабет у близьких" in warning
    # одне несуперечливе повідомлення: без «рахується тільки варіант (a)»
    assert not any("варіант (a)" in i.value for i in at.info)
