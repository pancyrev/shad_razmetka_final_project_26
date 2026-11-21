"""Freeze validation-selected settings, then evaluate each model once on HF test."""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from omegaconf import OmegaConf

from butterfly_bench.api import configure
from results.archive_paths import verify_frozen_files

MODELS = ("owlv2", "grounding_dino", "qwen", "gemma")
DATA_REVISION = "1316e3c7147d4399a03f90684e191aae4d798f09"
SOURCE_RUNS = {
    "owlv2": "results/experiments/validation_2026-11-13/owlv2_nms_0.5",
    "grounding_dino": "results/experiments/validation_2026-11-13/grounding_dino",
    "qwen": "results/experiments/prompt_search_2026-11-15/qwen/05_four_edges",
    "gemma": "results/experiments/prompt_search_2026-11-15/gemma/08_08_json_contract_first",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(campaign):
    """Persist all four choices before downloading or inspecting the test labels."""
    target = campaign / "frozen"
    target.mkdir(parents=True, exist_ok=False)
    for model in MODELS:
        previous = OmegaConf.load(ROOT / SOURCE_RUNS[model] / "config.yaml")
        preset = f"{model}_tuned" if model in {"qwen", "gemma"} else model
        cfg = configure([f"model={preset}", "data.split=test",
                         f"data.revision={DATA_REVISION}",
                         f"model.revision={previous.model.revision}",
                         f"postprocess.nms_iou={previous.postprocess.nms_iou}".replace("None", "null")],
                        output_dir=campaign / model)
        assert cfg.model == previous.model
        assert cfg.postprocess == previous.postprocess
        assert cfg.retries == previous.retries
        assert cfg.evaluation == previous.evaluation
        cfg.comet.experiment_name = f"test-400-final-{model}"
        cfg.comet.tags = [*cfg.comet.tags, "test-400", "final-config", "held-out-test"]
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
                       target / f"{model}.yaml")
    hashes = {str(path.relative_to(campaign)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in target.glob("*.yaml")}
    hashes.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (ROOT / "butterfly_bench").glob("*.py")})
    write_json(campaign / "plan.json", {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "expected_images": 400, "seed": 42, "source_runs": SOURCE_RUNS,
        "selection": "Settings selected on validation; no tuning on test.",
        "gpu_execution": "Sequential processes under /tmp/butterfly-prompt-search-gpu.lock",
        "sha256": hashes,
    })
    print("Frozen all four settings before test evaluation", flush=True)


def preflight(campaign):
    from huggingface_hub import hf_hub_download

    from butterfly_bench.data import load_samples

    verify_frozen_files(json.loads((campaign / "plan.json").read_text()), campaign)
    cfg = OmegaConf.load(campaign / "frozen/owlv2.yaml")
    samples, metadata = load_samples(cfg.data, cfg.mode, cfg.seed)
    assert len(samples) == 400
    validation = json.loads((ROOT / "results/experiments/validation_2026-11-13/qwen/data.json").read_text())
    assert not set(metadata["photo_ids"]) & set(validation["photo_ids"])
    for sample in samples:
        assert sample.truth is not None
    parquet = Path(hf_hub_download(cfg.data.repo_id, cfg.data.parquet_file, repo_type="dataset",
                                  revision=DATA_REVISION, token=False, local_files_only=True))
    metadata.update(parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
                    parquet_bytes=parquet.stat().st_size,
                    ground_truth_boxes=sum(len(s.truth) for s in samples),
                    negative_images=sum(not s.truth for s in samples),
                    no_overlap_with_validation=True)
    reference = json.loads((HERE / "dataset.json").read_text())
    for key in ("revision", "parquet_sha256", "photo_ids", "ground_truth_boxes", "negative_images"):
        if metadata[key] != reference[key]:
            raise ValueError(f"Evaluation data differs from the archived final dataset: {key}")
    write_json(campaign / "dataset.json", metadata)
    (campaign / "ground_truth.jsonl").write_text("".join(json.dumps({
        "photo_id": sample.photo_id, "boxes": sample.truth}, allow_nan=False) + "\n"
        for sample in samples))
    print(json.dumps({k: v for k, v in metadata.items() if k != "photo_ids"}), flush=True)


def run(model, campaign):
    verify_frozen_files(json.loads((campaign / "plan.json").read_text()), campaign)
    assert (campaign / "dataset.json").exists(), "Run preflight first"
    cfg = OmegaConf.load(campaign / "frozen" / f"{model}.yaml")
    cfg.output_dir = str(campaign / model)
    os.environ["COMET_OFFLINE_DIRECTORY"] = str(campaign / model / "comet-offline")
    from butterfly_bench.pipeline import run as run_pipeline

    result = run_pipeline(cfg)
    assert result["metrics"]["images"] == 400
    assert result["metrics"]["runtime_errors"] == 0, result
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", *MODELS))
    parser.add_argument("--campaign-dir", type=Path, default=HERE,
                        help="Directory for a new campaign, shared across all six actions")
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    if args.action == "prepare":
        prepare(campaign)
    elif args.action == "preflight":
        preflight(campaign)
    else:
        run(args.action, campaign)
