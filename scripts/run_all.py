"""Reproduce the submission without overwriting changed data or tuning the validator."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate_data import generate_cases, validate_cases, write_outputs
from src.frozen_evaluation import run_frozen_evaluation, normalized_sha256
from src.submission_reports import build_reports


def main() -> None:
    cases = generate_cases()
    validate_cases(cases)
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pit-regenerated-") as temp:
        stage = Path(temp)
        write_outputs(cases, stage)
        files = list(stage.glob("*.csv"))
        # Check every existing CSV before copying any missing one.
        for generated in files:
            existing = data_dir / generated.name
            if existing.exists() and normalized_sha256(existing) != normalized_sha256(generated):
                raise ValueError(f"{existing}: differs from seeded generation; refusing to replace data")
        for generated in files:
            if not (data_dir / generated.name).exists():
                shutil.copyfile(generated, data_dir / generated.name)
    print("[DATA] Seed 20261004: verified 300 cases (CRLF/LF permitted); existing CSVs preserved")
    (ROOT / "results").mkdir(exist_ok=True)
    lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "results/environment_lock.txt").write_text(lock, encoding="utf-8")
    run_frozen_evaluation()
    build_reports(ROOT)
    print("[DONE] Metrics, predictions, charts, failure analysis, README and four-slide outline generated")


if __name__ == "__main__":
    main()
