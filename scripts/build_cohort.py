"""Будує аналітичну вибірку: cohort.parquet, flow.json, variables.csv, cohort_notes.json."""

import argparse
import json
import logging
from dataclasses import asdict

import pandas as pd

from risk.config import CYCLES, FEATURE_LABELS_UK, MODULES, NHANES_CODES, PROCESSED_DIR, RAW_DIR
from risk.data.cohort import FlowStep, build_cohort, cohort_notes
from risk.data.download import file_name
from risk.data.harmonize import harmonize
from risk.data.load import load_cycle
from risk.data.weights import prevalence

log = logging.getLogger(__name__)


def variables_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """Довідник змінних: назва укр, код NHANES, одиниці, % пропусків у вибірці."""
    rows = [
        {
            "variable": var,
            "label_uk": FEATURE_LABELS_UK.get(var, var),
            "nhanes_code": code,
            "unit": unit,
            "missing_pct": round(cohort[var].isna().mean() * 100, 2),
        }
        for var, (code, unit) in NHANES_CODES.items()
    ]
    return pd.DataFrame(rows)


def check_raw() -> None:
    """Завершує скрипт із підказкою, якщо якогось XPT-файлу немає."""
    for cycle in CYCLES:
        for module in MODULES:
            name = f"{file_name(module, cycle)}.xpt"
            if not (RAW_DIR / name).exists():
                raise SystemExit(
                    f"Немає data/raw/{name}: спочатку `uv run python scripts/download_data.py`"
                )


def write_artifacts(cohort: pd.DataFrame, steps: list[FlowStep], notes: dict) -> None:
    """Записує артефакти вибірки в data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(PROCESSED_DIR / "cohort.parquet")
    flow = json.dumps([asdict(s) for s in steps], ensure_ascii=False, indent=2)
    (PROCESSED_DIR / "flow.json").write_text(flow, encoding="utf-8")
    variables_table(cohort).to_csv(PROCESSED_DIR / "variables.csv", index=False)
    text = json.dumps(notes, ensure_ascii=False, indent=2)
    (PROCESSED_DIR / "cohort_notes.json").write_text(text, encoding="utf-8")


def main() -> None:
    """Точка входу CLI."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_raw()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    harmonized = pd.concat([harmonize(load_cycle(c), c) for c in CYCLES], ignore_index=True)
    cohort, steps = build_cohort(harmonized)
    write_artifacts(cohort, steps, cohort_notes(harmonized, cohort))
    for s in steps:
        log.info(f"{s.n:>7}  (-{s.excluded:<6}) {s.label}")
    pd.set_option("display.width", 160)
    log.info(pd.concat([prevalence(cohort), prevalence(cohort, by="cycle")]).round(2).to_string())
    log.info(variables_table(cohort)[["variable", "missing_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
