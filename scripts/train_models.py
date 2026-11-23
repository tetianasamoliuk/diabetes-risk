"""Навчання logit / RF / HGB: models/{logit,rf,hgb}.joblib, metrics.json, test_predictions."""

import argparse
import time

import pandas as pd

from risk.config import MODELS_DIR, PROCESSED_DIR
from risk.models.train import evaluate_on_test, save, test_predictions_frame, train_all


def models_table(metrics: dict) -> pd.DataFrame:
    """CV (mean ± sd) і test-метрики трьох моделей."""
    rows = {
        name: {
            "cv_auc": f"{cv['cv_auc_mean']:.3f} ± {cv['cv_auc_sd']:.3f}",
            "test_auc": f"{t['auc']:.3f} ({t['auc_ci_low']:.3f}–{t['auc_ci_high']:.3f})",
            "brier": round(t["brier"], 4),
            "threshold": round(t["threshold"], 3),
            "sens": round(t["recall"], 3),
            "spec": round(t["specificity"], 3),
            "best_params": cv["best_params"],
        }
        for name, cv, t in ((n, metrics["cv"][n], metrics["test"][n]) for n in metrics["test"])
    }
    return pd.DataFrame(rows).T


def delong_table(metrics: dict) -> pd.DataFrame:
    """DeLong: пари моделей і найкраща модель vs FINDRISC (a)/(b)."""
    pairs = [{"pair": f"{r['a']} vs {r['b']}", **r} for r in metrics["delong"]]
    vs = [
        {"pair": f"{r['model']} vs FINDRISC ({r['variant']})", **r}
        for r in metrics["delong_vs_findrisc"]
    ]
    cols = ["pair", "auc_a", "auc_b", "diff", "ci_low", "ci_high", "z", "p"]
    return pd.DataFrame(pairs + vs)[cols].round(4)


def print_summary(metrics: dict, seconds: float) -> None:
    """Підсумок у консоль: розміри вибірок, метрики, порівняння AUC, FINDRISC, основна модель."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(
        f"N train {metrics['n_train']} (+{metrics['n_pos_train']}), "
        f"test {metrics['n_test']} (+{metrics['n_pos_test']}); {seconds:.0f} с"
    )
    print(models_table(metrics).to_string())
    print(delong_table(metrics).to_string(index=False))
    print(pd.DataFrame(metrics["findrisc"]).T.to_string())
    print(
        f"Найкраща модель ({metrics['best_model_rule']}): {metrics['best_model']}; "
        f"за test AUC (довідково): {metrics['test_auc_rank']}"
    )


def main() -> None:
    """Точка входу CLI."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    path = PROCESSED_DIR / "cohort.parquet"
    if not path.exists():
        raise SystemExit(
            "Немає data/processed/cohort.parquet: спочатку `uv run python scripts/build_cohort.py`"
        )
    cohort = pd.read_parquet(path)
    t0 = time.perf_counter()
    result = train_all(cohort)
    metrics = evaluate_on_test(result, cohort)
    save(result, metrics, test_predictions_frame(result, cohort), MODELS_DIR)
    print_summary(metrics, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
