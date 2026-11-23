"""CLI скриптів: `--help` друкує опис і нічого не запускає; зрозумілі помилки без вхідних даних."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WITH_HELP = [
    "train_models",
    "build_results",
    "download_data",
]
WATCHED = [
    ROOT / "models" / "metrics.json",
    ROOT / "data" / "processed" / "results_numbers.json",
]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mtimes() -> list[int | None]:
    return [p.stat().st_mtime_ns if p.exists() else None for p in WATCHED]


@pytest.mark.parametrize("name", WITH_HELP)
def test_help_prints_description_and_runs_nothing(name: str) -> None:
    before = _mtimes()
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("usage:")
    first_line = _load(name).__doc__.strip().splitlines()[0]
    assert first_line in run.stdout.replace("\n", " ") or first_line.split()[0] in run.stdout
    assert _mtimes() == before


def test_train_without_cohort_exits_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("train_models")
    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_models.py"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert "scripts/build_cohort.py" in str(exc.value)
    assert "data/processed/cohort.parquet" in str(exc.value)


def test_results_without_models_exits_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("build_results")
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in ("cohort.parquet", "flow.json", "variables.csv"):
        (processed / name).write_text("x")
    monkeypatch.setattr(module, "PROCESSED_DIR", processed)
    monkeypatch.setattr(module, "MODELS_DIR", tmp_path / "models")
    with pytest.raises(SystemExit) as exc:
        module.check_inputs()
    assert "scripts/train_models.py" in str(exc.value)


def test_results_without_cohort_exits_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("build_results")
    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)
    with pytest.raises(SystemExit) as exc:
        module.check_inputs()
    assert "scripts/build_cohort.py" in str(exc.value)
