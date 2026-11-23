"""Збирає таблиці й числа для документів з артефактів: results.xlsx та results_numbers.json."""

import argparse
import json
import time

import pandas as pd

from risk.config import MODELS_DIR, PROCESSED_DIR
from risk.data.cohort import FlowStep
from risk.external.ukraine import load_ukraine_stats
from risk.models.train import load_models, load_test_predictions
from risk.report.results import build_results, export_excel


def check_inputs() -> None:
    """Завершує скрипт із підказкою, якщо немає когорти чи навчених моделей."""
    for name in ("cohort.parquet", "flow.json", "variables.csv"):
        if not (PROCESSED_DIR / name).exists():
            raise SystemExit(
                f"Немає data/processed/{name}: спочатку `uv run python scripts/build_cohort.py`"
            )
    if not (MODELS_DIR / "metrics.json").exists() or not list(MODELS_DIR.glob("*.joblib")):
        raise SystemExit(
            "Немає models/*.joblib або metrics.json: "
            "спочатку `uv run python scripts/train_models.py`"
        )


def main() -> None:
    """Точка входу CLI."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_inputs()
    cohort = pd.read_parquet(PROCESSED_DIR / "cohort.parquet")
    raw = json.loads((PROCESSED_DIR / "flow.json").read_text(encoding="utf-8"))
    flow = [FlowStep(**r) for r in raw]
    variables = pd.read_csv(PROCESSED_DIR / "variables.csv")
    start = time.perf_counter()
    models, metrics = load_models()
    bundle = build_results(
        cohort, flow, variables,
        models=models, metrics=metrics, preds=load_test_predictions(),
        ukraine=load_ukraine_stats(),
    )  # fmt: skip
    print(f"build_results: {time.perf_counter() - start:.1f} с")
    (PROCESSED_DIR / "results.xlsx").write_bytes(export_excel(bundle))
    (PROCESSED_DIR / "results_numbers.json").write_text(
        json.dumps(bundle.numbers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"таблиць: {len(bundle.tables)}, рисунків: {len(bundle.figures)}")


if __name__ == "__main__":
    main()
