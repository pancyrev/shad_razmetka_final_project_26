"""Resolve paths recorded before the results directories were reorganized."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recorded_path(relative):
    path = Path(relative)
    if path.parts[:2] == ("model_results", "test_2026-11-16"):
        return ROOT.joinpath("results", "final_400", *path.parts[2:])
    if path.parts[:1] == ("model_results",):
        return ROOT.joinpath("results", "experiments", *path.parts[1:])
    return ROOT / path


def verify_frozen_files(plan, campaign_dir, historical_commit=None):
    """Verify the recorded campaign, or the live code when preparing a new run."""
    for relative, expected in plan["sha256"].items():
        path = campaign_dir / relative if Path(relative).parts[0] == "frozen" else recorded_path(relative)
        if historical_commit and Path(relative).parts[0] == "butterfly_bench":
            content = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{historical_commit}:{relative}"]
            )
        else:
            content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch for {relative}")
