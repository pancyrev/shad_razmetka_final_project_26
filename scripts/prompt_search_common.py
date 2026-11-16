"""Reproducible prompt experiments, serialized across both model workers."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("BUTTERFLY_PROMPT_SEARCH_RESULTS", ROOT / "model_results/prompt_search_train")).resolve()
BASELINE = ROOT / "model_results/validation_2026-11-13"
DATASET_REVISION = "863f76cfbcbcaf1563693b8a08e5128efa47a4c4"
LEGACY_DATASET_REVISION = "1316e3c7147d4399a03f90684e191aae4d798f09"
GPU_LOCK = Path("/tmp/butterfly-prompt-search-gpu.lock")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def event(model, variant, state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "gpu_schedule.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": datetime.now(UTC).isoformat(),
                                 "pid": os.getpid(), "model": model,
                                 "variant": variant, "state": state}) + "\n")


def verify_training_config(cfg):
    if (cfg.data.split != "train" or cfg.data.revision != DATASET_REVISION
            or cfg.data.parquet_file != "data/train-00000-of-00001.parquet"):
        raise ValueError("Prompt search requires the pinned train split; use a new results directory")


@contextmanager
def gpu_lock(model, variant):
    """Hold one shared advisory lock through child exit and server cleanup."""
    print(f"{model}/{variant}: waiting for exclusive GPU", flush=True)
    with GPU_LOCK.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        event(model, variant, "acquired")
        try:
            yield handle.fileno()
        finally:
            event(model, variant, "released")
            fcntl.flock(handle, fcntl.LOCK_UN)


def stop_group(process):
    """Also terminate a surviving llama-server descendant before unlocking."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
    # The inference subprocess starts its own group; no unrelated jobs belong here.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_search(model, prompts):
    if model not in {"qwen", "gemma"} or len(prompts) != 10:
        raise ValueError("Expected qwen/gemma and exactly 10 prompt candidates")
    if len({p["prompt"] for p in prompts}) != 10:
        raise ValueError("All ten prompt texts must differ")
    if len({p["name"] for p in prompts}) != 10 or any(
        not re.fullmatch(r"[a-zA-Z0-9_-]+", p["name"]) for p in prompts
    ):
        raise ValueError("Prompt names must be unique filename-safe identifiers")
    os.chdir(ROOT)
    target = RESULTS / model
    target.mkdir(parents=True, exist_ok=True)
    prompt_file = target / "prompts.json"
    if prompt_file.exists() and json.loads(prompt_file.read_text()) != prompts:
        raise ValueError("Saved prompts differ; do not overwrite an existing search")
    write_json(prompt_file, prompts)
    for index, variant in enumerate(prompts):
        out = target / f"{index + 1:02d}_{variant['name']}"
        if (out / "metrics.json").exists():
            from omegaconf import OmegaConf

            verify_training_config(OmegaConf.load(out / "config.yaml"))
            print(f"Already evaluated: {out}", flush=True)
            continue
        if (out / "status.json").exists():
            raise FileExistsError(f"Incomplete previous run at {out}; inspect before resuming")
        out.mkdir(parents=True, exist_ok=True)
        with gpu_lock(model, variant["name"]) as lock_fd:
            print(f"Running {model} {index + 1}/10: {variant['name']}", flush=True)
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            command = [sys.executable, str(Path(__file__).resolve()), "--run-one",
                       "--model", model, "--index", str(index), "--output", str(out)]
            with (out / "run.log").open("w") as log_handle:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lock_fd,))
                try:
                    returncode = process.wait()
                finally:
                    stop_group(process)
            if returncode:
                raise RuntimeError(f"Run failed ({returncode}); inspect {out / 'run.log'}")
        metrics = json.loads((out / "metrics.json").read_text())
        print(json.dumps({"model": model, "variant": variant["name"],
                          "f1_05": metrics["iou_0.5/f1"],
                          "f1_075": metrics["iou_0.75/f1"],
                          "processing_errors": metrics["processing_errors"]}), flush=True)
    return summarize(model, prompts)


def summarize(model, prompts):
    from omegaconf import OmegaConf

    rows = []
    for index, prompt in enumerate(prompts):
        out = RESULTS / model / f"{index + 1:02d}_{prompt['name']}"
        if not (out / "metrics.json").exists():
            continue
        metrics = json.loads((out / "metrics.json").read_text())
        saved_cfg = OmegaConf.load(out / "config.yaml")
        verify_training_config(saved_cfg)
        actual = saved_cfg.model.prompt
        if actual != prompt["prompt"] or metrics["images"] != 100:
            raise ValueError(f"Prompt or image count mismatch in {out}")
        rows.append({"name": prompt["name"], "output_dir": str(out.relative_to(ROOT)),
                     "prompt_sha256": hashlib.sha256(actual.encode()).hexdigest(),
                     **metrics})
    rows.sort(key=lambda r: (r["iou_0.5/f1"], r["iou_0.75/f1"],
                             -r["processing_errors"]), reverse=True)
    write_json(RESULTS / model / "summary.json", rows)
    if rows:
        best = next(p for p in prompts if p["name"] == rows[0]["name"])
        write_json(RESULTS / model / "best_prompt.json", best)
        (RESULTS / model / "best_prompt.txt").write_text(best["prompt"] + "\n")
    return rows


def cached_tuning(cfg, mode, seed):
    """Read 100 train images, or the explicitly pinned historical validation split."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from butterfly_bench.data import Sample
    from butterfly_bench.schema import validate_boxes

    legacy = cfg.split == "validation" and cfg.revision == LEGACY_DATASET_REVISION
    if mode != "test" or not (cfg.split == "train" or legacy) or cfg.limit is not None or cfg.shuffle:
        raise ValueError("Search requires all 100 train images in original order")
    if cfg.photo_ids is not None:
        raise ValueError("Subset selection is forbidden for this search")
    path = Path(hf_hub_download(cfg.repo_id, cfg.parquet_file, repo_type="dataset",
                               revision=cfg.revision, token=False, local_files_only=legacy))
    rows = pq.read_table(path).to_pylist()
    metadata = json.loads((BASELINE / "qwen/data.json").read_text())
    if [r["photo_id"] for r in rows] != metadata["photo_ids"] or len(rows) != 100:
        raise ValueError("Tuning data differs from original 100 image benchmark")
    reference = {r["photo_id"]: r["boxes"] for r in map(json.loads, (
        ROOT / "annotations/worker_reference_first_pass.jsonl").read_text().splitlines())}
    samples = []
    for row in rows:
        if row["boxes"] != reference[row["photo_id"]]:
            raise ValueError(f"Ground truth changed for {row['photo_id']}")
        image = row["image"]
        payload = ({"bytes": image["bytes"]} if image.get("bytes") is not None
                   else {"path": image["path"]})
        samples.append(Sample(row["photo_id"], payload, validate_boxes(row["boxes"])))
    metadata.update(revision=cfg.revision, split=cfg.split, parquet_file=cfg.parquet_file,
                    count=len(samples), local_parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return samples, metadata


# Archived experiment drivers import this name and pin the old dataset revision.
cached_validation = cached_tuning


def run_one(model, index, output):
    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    from butterfly_bench import pipeline

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("COMET_OFFLINE_DIRECTORY", str(Path(output).resolve() / "comet-offline"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    prompts = json.loads((RESULTS / model / "prompts.json").read_text())
    prompt = prompts[index]
    # Reuse model settings and the same 100 images under their new pinned train name.
    cfg = OmegaConf.load(BASELINE / model / "config.yaml")
    cfg.data.revision = DATASET_REVISION
    cfg.data.split = "train"
    cfg.data.parquet_file = "data/train-00000-of-00001.parquet"
    verify_training_config(cfg)
    cfg.output_dir = str(Path(output).resolve())
    cfg.model.prompt = prompt["prompt"]
    cfg.comet.experiment_name = f"prompt-search-{model}-{index + 1:02d}-{prompt['name']}"
    cfg.comet.tags = [*cfg.comet.tags, "prompt-search", "train-tuning"]
    if cfg.seed != 42 or cfg.mode != "test":
        raise ValueError("Unexpected baseline configuration")
    # Metadata resolution only is replaced; actual model, parser, retries, metrics and tracking are unchanged.
    with patch.object(pipeline, "resolve_model", lambda config: config.revision), patch.object(
        pipeline, "load_samples", cached_tuning
    ):
        result = pipeline.run(cfg)
    if result["metrics"]["images"] != 100:
        raise ValueError("Incomplete train evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-one", action="store_true", required=True)
    parser.add_argument("--model", choices=["qwen", "gemma"], required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_one(args.model, args.index, args.output)
