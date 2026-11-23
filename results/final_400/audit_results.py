"""Verify the fixed test campaign from saved inputs, attempts and predictions."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from butterfly_bench.metrics import evaluate, nms
from butterfly_bench.models import prompt_for
from butterfly_bench.schema import (
    PredictionError,
    detector_record,
    parse_jsonl_response,
    validate_record,
)
from results.archive_paths import recorded_path, verify_frozen_files


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def audit_model(model, dataset, truth, plan, campaign):
    directory = campaign / model
    cfg = OmegaConf.load(directory / "config.yaml")
    frozen = OmegaConf.load(campaign / "frozen" / f"{model}.yaml")
    assert cfg == frozen, model
    source = OmegaConf.load(recorded_path(plan["source_runs"][model]) / "config.yaml")
    assert cfg.model == source.model
    for field in ("seed", "deterministic", "mode", "retries", "postprocess", "evaluation"):
        assert cfg[field] == source[field], (model, field)
    actual_data = read_json(directory / "data.json")
    assert all(actual_data[key] == dataset[key] for key in actual_data)
    records = read_lines(directory / "predictions.jsonl")
    assert [record["photo_id"] for record in records] == dataset["photo_ids"]
    assert len(records) == 400
    predictions = {record["photo_id"]: record for record in records}
    attempts = read_lines(directory / "attempts.jsonl")
    grouped = defaultdict(list)
    for entry in attempts:
        grouped[entry["photo_id"]].append(entry)
    assert set(grouped) == set(predictions)
    timings = read_lines(directory / "timings.jsonl")
    assert [entry["photo_id"] for entry in timings] == dataset["photo_ids"]
    for timing in timings:
        photo_id = timing["photo_id"]
        image_attempts = grouped[photo_id]
        assert len(image_attempts) == timing["attempts"] <= cfg.retries.max_attempts
        feedback = None
        for index, entry in enumerate(image_attempts, 1):
            assert entry["attempt"] == index
            assert entry["seed"] == (cfg.seed + photo_id * 1009 + index - 1) % 2**32
            if model in {"qwen", "gemma"} and "raw_text" in entry:
                assert entry["details"]["effective_prompt"] == prompt_for(cfg.model, photo_id, feedback)
            try:
                if entry.get("raw_record") is None:
                    parsed = parse_jsonl_response(entry.get("raw_text"), photo_id)
                    scores = [1.0] * len(parsed["boxes"])
                else:
                    parsed = validate_record(entry["raw_record"], photo_id)
                    scores = entry["details"]["scores"]
                    rebuilt, clipped = detector_record(
                        photo_id, entry["details"]["raw_boxes_xyxy"], scores,
                        entry["image_size"], clip=cfg.model.clip_boxes)
                    assert rebuilt == parsed and clipped == entry["details"]["clipped_boxes"]
                parsed["boxes"], filtered_scores = nms(parsed["boxes"], scores, cfg.postprocess.nms_iou)
            except PredictionError as exc:
                assert entry["stage"] == exc.stage
                feedback = entry["error"]
            else:
                assert entry["stage"] == "success" and index == len(image_attempts)
                assert parsed == predictions[photo_id]
                assert filtered_scores == timing["scores"]
        final = predictions[photo_id]
        if image_attempts[-1]["stage"] == "success":
            validate_record(final, photo_id)
        else:
            assert len(image_attempts) == cfg.retries.max_attempts
            assert final == {"photo_id": photo_id, "boxes": None, "error": feedback}
        for stage in ("parsing", "validation", "runtime"):
            assert timing[f"{stage}_errors"] == sum(a["stage"] == stage for a in image_attempts)
    metrics = read_json(directory / "metrics.json")
    recomputed = evaluate(predictions, truth, cfg.evaluation.iou_thresholds, cfg.evaluation.primary_iou)
    assert all(metrics[key] == value for key, value in recomputed.items())
    assert close(metrics["processing_seconds"], sum(t["seconds"] for t in timings))
    assert metrics["total_attempts"] == len(attempts)
    for stage in ("parsing", "validation", "runtime"):
        assert metrics[f"{stage}_errors"] == sum(a["stage"] == stage for a in attempts)
    assert metrics["runtime_errors"] == 0
    assert (directory / "submission.jsonl").exists() == (metrics["processing_errors"] == 0)
    status = read_json(directory / "status.json")["status"]
    assert status == ("completed_with_errors" if metrics["processing_errors"] else "complete")
    model_info = read_json(directory / "model.json")
    assert model_info["revision"] == cfg.model.revision
    if model == "gemma":
        assert model_info["quantization"] == "UD-Q4_K_XL"
        assert model_info["gpu_layers_offloaded"] == 49
    else:
        assert model_info["device"] == "cuda:0" and not model_info["quantized"]
        assert model_info["dtype"] == ("torch.bfloat16" if model == "qwen" else "torch.float32")
    gpu = read_lines(directory / "gpu.jsonl")
    assert gpu and all(g["name"] == "NVIDIA GeForce RTX 3080" for g in gpu)
    return {"model": model, "status": status, "images": 400,
            "configuration_matches_validation_selection_and_frozen_plan": True,
            "requests_seeds_retries_parsing_geometry_nms_and_metrics_verified": True,
            "gpu_first_timestamp": gpu[0]["timestamp"], "gpu_last_timestamp": gpu[-1]["timestamp"],
            "metrics": metrics, "comet": read_json(directory / "comet.json"),
            "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ("config.yaml", "predictions.jsonl", "attempts.jsonl", "metrics.json")}}


def main(campaign, partial, output):
    plan = read_json(campaign / "plan.json")
    historical = any(name.startswith("model_results/") for name in plan["sha256"])
    source_commit = read_json(campaign / "owlv2/environment.json")["git_commit"] if historical else None
    verify_frozen_files(plan, campaign, historical_commit=source_commit)
    dataset = read_json(campaign / "dataset.json")
    truth_rows = read_lines(campaign / "ground_truth.jsonl")
    truth = {row["photo_id"]: row["boxes"] for row in truth_rows}
    assert list(truth) == dataset["photo_ids"] and len(truth) == 400
    validation = read_json(ROOT / "results/experiments/validation_2026-11-13/qwen/data.json")
    assert not set(truth) & set(validation["photo_ids"])
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet = Path(hf_hub_download(dataset["repo_id"], dataset["parquet_file"], repo_type="dataset",
                                  revision=dataset["revision"], token=False, local_files_only=True))
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == dataset["parquet_sha256"]
    assert pq.read_table(parquet, columns=["photo_id", "boxes"]).to_pylist() == truth_rows
    rows = []
    for model in plan["source_runs"]:
        status_path = campaign / model / "status.json"
        if partial and (not status_path.exists() or read_json(status_path)["status"] == "running"):
            continue
        rows.append(audit_model(model, dataset, truth, plan, campaign))
    for previous, current in pairwise(rows):
        assert previous["gpu_last_timestamp"] < current["gpu_first_timestamp"]
    result = {"audited_at_utc": datetime.now(UTC).isoformat(), "complete": len(rows) == 4,
              "test_ids_disjoint_from_validation": True,
              "ground_truth_matches_pinned_hf_parquet": True,
              "gpu_runs_sequential_without_overlap": True, "runs": rows}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        m = row["metrics"]
        print(json.dumps({"model": row["model"], "f1_05": m["iou_0.5/f1"],
                          "f1_075": m["iou_0.75/f1"], "processing_errors": m["processing_errors"],
                          "parsing_errors": m["parsing_errors"], "validation_errors": m["validation_errors"],
                          "seconds": m["processing_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--campaign-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.campaign_dir, args.partial, args.output or args.campaign_dir / "audit.json")
