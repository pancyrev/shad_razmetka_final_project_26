"""Read back Comet metrics and assets; never upload or modify experiments."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from comet_ml.api import API

FILES = ("config.yaml", "environment.json", "data.json", "model.json", "metrics.json",
         "predictions.jsonl", "timings.jsonl", "gpu.jsonl", "attempts.jsonl")
api = API(cache=False)
rows = []
for model in ("owlv2", "grounding_dino", "qwen", "gemma"):
    directory = HERE / model
    info = json.loads((directory / "comet.json").read_text())
    result = {"model": model, "url": info.get("url"), "mode": info["mode"]}
    try:
        assert info["mode"] == "online"
        workspace, project, _ = urlparse(info["url"]).path.strip("/").split("/")
        experiment = api.get_experiment(workspace, project, info["experiment_key"])
        local_metrics = json.loads((directory / "metrics.json").read_text())
        cloud_metrics = {m["name"]: m["valueCurrent"] for m in experiment.get_metrics_summary()}
        mismatches = [key for key, value in local_metrics.items()
                      if key not in cloud_metrics or not math.isclose(
                          float(cloud_metrics[key]), value, rel_tol=1e-9, abs_tol=1e-9)]
        result["metric_mismatches"] = mismatches
        assets = {a["fileName"]: a for a in experiment.get_asset_list(timeout=30)}
        result["missing_assets"] = [name for name in FILES if name not in assets]
        hashes = {}
        for name in FILES:
            if name in assets:
                content = experiment.get_asset(assets[name]["assetId"], return_type="binary")
                hashes[name] = {"sha256": hashlib.sha256(content).hexdigest(),
                                "matches_local": content == (directory / name).read_bytes()}
        result["assets"] = hashes
        parameter = experiment.get_parameters_summary("model.prompt")
        parameter_value = parameter.get("valueCurrent") if parameter else None
        cfg = OmegaConf.load(directory / "config.yaml")
        result["full_prompt_parameter_matches"] = parameter_value == cfg.model.prompt
        result["all_metrics_verified"] = not mismatches
        result["all_nine_assets_verified_byte_for_byte"] = (
            len(hashes) == len(FILES) and all(item["matches_local"] for item in hashes.values()))
    except Exception as exc:  # noqa: BLE001 -- report unavailable cloud data without losing local results
        result["verification_error_type"] = type(exc).__name__
        result["all_metrics_verified"] = False
        result["all_nine_assets_verified_byte_for_byte"] = False
    rows.append(result)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}), flush=True)
report = {"verified_at_utc": datetime.now(UTC).isoformat(), "experiments": rows}
(HERE / "comet_verification.json").write_text(json.dumps(report, indent=2) + "\n")
