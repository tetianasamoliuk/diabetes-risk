"""Схеми для документів: числа з артефактів прогону → 9 PNG (300 dpi).

За замовчуванням PNG пишуться у docs/guide/img і docs/thesis/img; інші теки — через --out.
"""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

from risk.config import (
    ACTIVE_MIN_PER_WEEK,
    HBA1C_THRESHOLD,
    MODEL_NAMES_UK,
    MODELS_DIR,
    PROCESSED_DIR,
    ROOT,
)
from risk.data.preprocessing import PREPROCESSING_STEPS
from risk.findrisc.score import ORIGINAL_VS_MODIFIED
from risk.viz import schemas
from risk.viz.style import close, fig_to_png

NBSP = schemas.NBSP

OUT_DIRS = [ROOT / "docs" / "guide" / "img", ROOT / "docs" / "thesis" / "img"]
APP_PAGES = ROOT / "app" / "pages"
SHAP_MODEL = "hgb"  # сторінка «Оцінка ризику» пояснює SHAP-ом саме Gradient Boosting
STATUSES = {str(i): "реалізовано" for i in range(1, 10)}
PROJECT_MAP_ROWS = (
    (
        "Дані NHANES",
        "pandas, pyarrow",
        "src/risk/data\n(download, load)\nscripts/download_data.py",
        "стор. «Дані і вибірка»",
    ),
    (
        "Відбір і підготовка",
        "pandas, numpy",
        "src/risk/data\n(harmonize, cohort, preprocessing, weights)\nscripts/build_cohort.py",
        "стор. «Дані і вибірка»",
    ),
    (
        "Статистика й фактори ризику",
        "scipy, statsmodels",
        "src/risk/stats",
        "стор. «Статистика», «Фактори ризику»",
    ),
    (
        "Навчання моделей",
        "scikit-learn, joblib",
        "src/risk/models\n(pipelines, train)\nscripts/train_models.py",
        "стор. «Моделі»",
    ),
    (
        "Порівняння моделей і FINDRISC",
        "scikit-learn; власний код тесту ДеЛонга і шкали",
        "src/risk/models\n(evaluate, delong, summary, calibration, threshold)\nsrc/risk/findrisc",
        "стор. «Моделі», «FINDRISC»",
    ),
    (
        "Пояснення результату",
        "shap",
        "src/risk/models/explain.py\nsrc/risk/interpret.py\nsrc/risk/person.py",
        "стор. «Моделі», «Оцінка ризику»",
    ),
    (
        "Порівняння з Україною",
        "pandas",
        "src/risk/external\ndata/external",
        "стор. «Порівняння з Україною»",
    ),
    (
        "Графіки й схеми",
        "matplotlib",
        "src/risk/viz\nscripts/build_schemas.py",
        "усі сторінки; документи",
    ),
    ("Застосунок", "streamlit", "app/", "сторінки застосунку в браузері"),
    (
        "Експорт",
        "openpyxl",
        "src/risk/report\nscripts/build_results.py",
        "стор. «Експорт»: Excel і ZIP",
    ),
    (
        "Запуск і розгортання",
        "uv, Docker",
        "pyproject.toml, uv.lock, Makefile, Dockerfile, docker-compose.yml",
        "адреса застосунку",
    ),
    ("Перевірка якості", "pytest, ruff", "tests/", "—"),
)  # (етап, чим зроблено, де в коді, де це видно)
BLOCK_REFS = {
    "1": (
        "Табл. 1.1–1.3; Рис. 1.1–1.2; завантаження власного файлу CSV/Excel "
        "(шаблон — табл. 1.5, лише в застосунку)"
    ),
    "2": "Табл. 1.4 «Попередня обробка»; імпутація і стандартизація — всередині конвеєра моделей",
    "3": "Табл. 2.1–2.4, 6.1–6.2; Рис. 2.1–2.6, 6.1",
    "4": "Табл. 3.1–3.7; Рис. 3.1–3.2",
    "5": "Табл. 4.1 (коефіцієнти логіту); Рис. 4.6–4.7 (SHAP)",
    "6": "Табл. 5.1–5.6; Рис. 5.1–5.2",
    "7": "Табл. 4.2–4.5; Рис. 4.1–4.5",
    "8": "Ймовірності моделей, бал FINDRISC, SHAP, сценарії «що якщо»",
    "9": "Excel з усіма таблицями, ZIP рисунків",
}  # номер блоку керівника → ключові таблиці й рисунки застосунку
OUT_OF_SCOPE = [
    "харчування",
    "тригліцериди",
    "глюкоза як фактор",
    "CANRISK",
    "моделі метаболічного синдрому",
    "PDF-звіт (замість нього — Excel)",
]
TOTAL_ROW = "Максимум балів"
REMOVED = "виключено"
GLUCOSE_ITEM = "Висока глюкоза в анамнезі"
FINDRISC_NOTES = [
    "Спадковість: лише найближчі родичі (батьки, брати, сестри).",
    f"Активність: GPAQ ≥{NBSP}{ACTIVE_MIN_PER_WEEK} хв/тиждень замість «≥{NBSP}30 хв щодня».",
]


def _read_text(path: Path) -> str:
    """Читає артефакт; без нього завершує скрипт із підказкою."""
    if not path.exists():
        raise SystemExit(
            f"Немає {path}: спочатку `uv run python scripts/build_cohort.py` "
            "і `uv run python scripts/train_models.py`"
        )
    return path.read_text(encoding="utf-8")


def load_artifacts() -> tuple[list[dict], dict, dict, dict]:
    """flow.json, metrics.json, results_numbers.json і необов'язковий cohort_notes.json."""
    notes = PROCESSED_DIR / "cohort_notes.json"
    return (
        json.loads(_read_text(PROCESSED_DIR / "flow.json")),
        json.loads(_read_text(MODELS_DIR / "metrics.json")),
        json.loads(_read_text(PROCESSED_DIR / "results_numbers.json")),
        json.loads(_read_text(notes)) if notes.exists() else {},
    )


def _points(text: str) -> str:
    """«0/2/3/4 (<45; …)» → «0/2/3/4»."""
    return text.split(" ")[0]


def _max(points: str) -> int:
    """Найбільший бал із рядка «0/2/3»."""
    return max(int(p) for p in points.split("/"))


def findrisc_items() -> tuple[list[tuple[str, str]], list[tuple[str, str]], int, tuple[int, int]]:
    """Пункти оригіналу й модифікації та максимуми балів — з таблиці ORIGINAL_VS_MODIFIED."""
    rows = [r for r in ORIGINAL_VS_MODIFIED if r["item"] != TOTAL_ROW]
    original = [(r["item"], r["original"]) for r in rows]
    modified = [(r["item"], _points(r["modified"])) for r in rows if r["modified"] != REMOVED]
    max_mod = sum(_max(p) for _, p in modified)
    glucose = _max(dict(modified)[GLUCOSE_ITEM])
    return original, modified, sum(_max(p) for _, p in original), (max_mod, max_mod - glucose)


def findrisc_variant_rows(metrics: dict, n_items: int, glucose_points: int) -> list[dict]:
    """Стрічки a/b/c: хто входить, пункт про глюкозу, ціль; N, AUC і придатність — з metrics."""
    hba1c = f"HbA1c{NBSP}≥{NBSP}{HBA1C_THRESHOLD:g}{NBSP}%"
    texts = {
        "a": ("усі з повною анкетою", f"ні ({n_items - 1} пунктів)",
              f"ЦД2: діагноз або {hba1c}"),
        "b": ("лише без діагностованого діабету", "так", f"ЦД2 без діагнозу: {hba1c}"),
        "c": ("усі з повною анкетою", f"так; діагностованим автоматично {glucose_points} балів",
              f"ЦД2: діагноз або {hba1c}"),
    }  # fmt: skip
    fr = metrics["findrisc"]
    return [
        {"key": k, "who": who, "glucose": glu, "target": target, "valid": fr[k]["valid"],
         "n": fr[k]["n"], "auc": fr[k]["auc"]}
        for k, (who, glu, target) in texts.items()
    ]  # fmt: skip


def _excluded(flow: list[dict], needle: str) -> int | None:
    """Скільки виключено на кроці, у назві якого є `needle`."""
    return next((s["excluded"] for s in flow if needle in s["label"]), None)


def _split_figure(flow: list[dict], metrics: dict, numbers: dict) -> Figure:
    """Схема поділу даних на навчальну й тестову вибірки."""
    fr = metrics["findrisc"]
    return schemas.data_split(
        flow[-1]["n"], metrics["n_train"], metrics["n_test"],
        numbers.get("n_multivariable"), numbers.get("n_logit_train"),
        fr["a"]["n"], fr["b"]["n"], fr["b"]["n_pos"],
    )  # fmt: skip


def _findrisc_figure(metrics: dict) -> Figure:
    """Схема варіантів FINDRISC."""
    original, modified, max_original, max_modified = findrisc_items()
    glucose = _max(dict(modified)[GLUCOSE_ITEM])
    rows = findrisc_variant_rows(metrics, len(modified), glucose)
    return schemas.findrisc_variants(
        original, modified, max_original, max_modified, rows, FINDRISC_NOTES
    )


def build_figures(flow: list[dict], metrics: dict, numbers: dict, notes: dict) -> dict[str, Figure]:
    """Ім'я файлу → схема."""
    best = metrics["best_model"]
    n_cohort = flow[-1]["n"]
    return {
        "схема_9_блоків.png": schemas.blocks_map(STATUSES, BLOCK_REFS, OUT_OF_SCOPE),
        "схема_архітектури.png": schemas.architecture(1 + len(list(APP_PAGES.glob("*.py")))),
        "схема_моделювання.png": schemas.modelling_pipeline(
            n_cohort, metrics["n_train"], metrics["n_test"],
            len(metrics["cv"][best]["cv_folds"]), MODEL_NAMES_UK[best],
        ),
        "схема_поділу_даних.png": _split_figure(flow, metrics, numbers),
        "схема_цільової_змінної.png": schemas.target_tree(
            flow[-1]["excluded"], _excluded(flow, "HbA1c"), notes.get("t1d_proxy_n")
        ),
        "схема_findrisc.png": _findrisc_figure(metrics),
        "схема_оцінки_людини.png": schemas.person_flow(
            [MODEL_NAMES_UK[k] for k in metrics["test"]], MODEL_NAMES_UK[SHAP_MODEL]
        ),
        "схема_обробки_даних.png": schemas.preprocessing_flow(PREPROCESSING_STEPS),
        "схема_проєкту.png": schemas.project_map(PROJECT_MAP_ROWS),
    }  # fmt: skip


def save(figures: dict[str, Figure], out_dirs: list[Path]) -> list[Path]:
    """PNG 300 dpi у кожну теку; фігури закриваються."""
    paths = []
    for name, fig in figures.items():
        png = fig_to_png(fig, dpi=300, tight=False)  # ширина PNG = полотно: масштаб у друці відомий
        close(fig)
        for d in out_dirs:
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(png)
            paths.append(d / name)
    return paths


def main(argv: list[str] | None = None) -> None:
    """Точка входу CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", nargs="+", type=Path, default=OUT_DIRS, help="теки для PNG")
    args = parser.parse_args(argv)
    flow, metrics, numbers, notes = load_artifacts()
    if numbers.get("n_logit_train") is None:
        print("Увага: у results_numbers.json немає n_logit_train — N для табл. 4.1 не показано")
    if notes.get("t1d_proxy_n") is None:
        print("Увага: немає cohort_notes.json з t1d_proxy_n — число ймовірних ЦД1 не показано")
    paths = save(build_figures(flow, metrics, numbers, notes), args.out)
    for d in args.out:
        print(f"Збережено {sum(p.parent == d for p in paths)} схем у {d}")


if __name__ == "__main__":
    main()
