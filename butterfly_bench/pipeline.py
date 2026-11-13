import json
import logging
import os
import random
import time
from pathlib import Path

from omegaconf import OmegaConf

from .data import load_samples, open_sample
from .metrics import evaluate, nms
from .models import create_backend, resolve_model
from .schema import PredictionError, parse_jsonl_response, validate_record
from .tracking import CometLogger, GPUTrace, environment_info, write_json

log = logging.getLogger(__name__)


def seed_everything(seed, deterministic=True):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def run_sample(backend, sample, cfg, attempts_handle):
    """Retry malformed/invalid responses; never convert a failure to empty success."""
    started = time.perf_counter()
    feedback = None
    parsing_errors = validation_errors = runtime_errors = 0
    last_error = None
    for attempt in range(1, cfg.retries.max_attempts + 1):
        attempt_start = time.perf_counter()
        attempt_seed = (cfg.seed + sample.photo_id * 1009 + attempt - 1) % (2**32)
        entry = {"photo_id": sample.photo_id, "attempt": attempt, "seed": attempt_seed}
        try:
            image = open_sample(sample)
            entry["image_size"] = list(image.size)
            output = backend.predict(image, sample.photo_id, attempt=attempt,
                                     seed=attempt_seed, feedback=feedback)
            entry.update(raw_text=output.text, raw_record=output.record, details=output.details)
            record = (parse_jsonl_response(output.text, sample.photo_id)
                      if output.record is None else validate_record(output.record, sample.photo_id))
            scores = output.scores if output.record is not None else [1.0] * len(record["boxes"])
            record["boxes"], scores = nms(record["boxes"], scores, cfg.postprocess.nms_iou)
            entry.update(stage="success", seconds=time.perf_counter() - attempt_start)
            attempts_handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            attempts_handle.flush()
            return record, {"photo_id": sample.photo_id, "attempts": attempt,
                            "parsing_errors": parsing_errors, "validation_errors": validation_errors,
                            "runtime_errors": runtime_errors, "scores": scores,
                            "seconds": time.perf_counter() - started}
        except Exception as exc:  # noqa: BLE001 -- per-image failures must be recorded and retried
            stage = exc.stage if isinstance(exc, PredictionError) else "runtime"
            parsing_errors += stage == "parsing"
            validation_errors += stage == "validation"
            runtime_errors += stage == "runtime"
            last_error = f"{type(exc).__name__}: {exc}"
            feedback = last_error
            entry.update(stage=stage, error=last_error, seconds=time.perf_counter() - attempt_start)
            # Nonfinite values are kept as a diagnostic string, never as invalid JSON.
            attempts_handle.write(json.dumps(_safe_json(entry), ensure_ascii=False, allow_nan=False) + "\n")
            attempts_handle.flush()
            log.warning("photo_id=%s attempt=%s %s: %s", sample.photo_id, attempt, stage, exc)
            if "out of memory" in str(exc).lower():
                import torch

                torch.cuda.empty_cache()
    return {"photo_id": sample.photo_id, "boxes": None, "error": last_error}, {
        "photo_id": sample.photo_id, "attempts": cfg.retries.max_attempts,
        "parsing_errors": parsing_errors, "validation_errors": validation_errors,
        "runtime_errors": runtime_errors, "scores": [], "seconds": time.perf_counter() - started}


def _safe_json(value):
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def run(cfg):
    if cfg.mode not in {"test", "inference"}:
        raise ValueError("mode must be test or inference")
    if cfg.retries.max_attempts < 1:
        raise ValueError("retries.max_attempts must be at least 1")
    thresholds = list(cfg.evaluation.iou_thresholds)
    if cfg.evaluation.primary_iou not in thresholds or any(not 0 < v <= 1 for v in thresholds):
        raise ValueError("IoU thresholds must be in (0,1] and include primary_iou")
    nms_threshold = cfg.postprocess.nms_iou
    if nms_threshold is not None and not 0 <= nms_threshold <= 1:
        raise ValueError("postprocess.nms_iou must be in [0,1] or null")
    # Initialize SDK before torch; experiment creation still follows data validation.
    os.environ.setdefault("COMET_TIMEOUT_UPLOAD", "60")
    os.environ.setdefault("COMET_TIMEOUT_CLEANING", "30")
    import comet_ml  # noqa: F401

    seed_everything(cfg.seed, cfg.deterministic)
    import torch

    if not torch.cuda.is_available() or not cfg.model.device.startswith("cuda"):
        raise RuntimeError("A working CUDA GPU is required; CPU fallback is disabled")
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "predictions.jsonl").exists() or (output_dir / "status.json").exists():
        raise FileExistsError(f"run artifacts already exist in {output_dir}; choose a new output_dir")
    config = OmegaConf.to_container(cfg, resolve=True)
    # Pin actual hub commits in saved config before constructing processors/models.
    revision = resolve_model(cfg.model)
    config["model"]["revision"] = revision
    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
    env = environment_info()
    env.update(gpu=torch.cuda.get_device_name(cfg.model.device), cuda=torch.version.cuda,
               seed=cfg.seed, deterministic=cfg.deterministic)
    write_json(output_dir / "environment.json", env)
    write_json(output_dir / "status.json", {"status": "running"})
    backend = logger = None
    trace = GPUTrace(output_dir, int(cfg.model.device.split(":")[-1]))
    trace.start()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(cfg.model.device)
    try:
        data_started = time.perf_counter()
        samples, data_metadata = load_samples(cfg.data, cfg.mode, cfg.seed)
        data_seconds = time.perf_counter() - data_started
        write_json(output_dir / "data.json", data_metadata)
        config["data"]["revision"] = data_metadata.get("revision", cfg.data.get("revision"))
        OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
        logger = CometLogger(cfg.comet, output_dir, config)
        model_started = time.perf_counter()
        backend = create_backend(cfg.model, revision, output_dir)
        load_seconds = time.perf_counter() - model_started
        write_json(output_dir / "model.json", backend.metadata)
        predictions, timings = {}, []
        with (output_dir / "predictions.jsonl").open("w") as pred_file, (
            output_dir / "attempts.jsonl").open("w") as attempts_file, (
            output_dir / "timings.jsonl").open("w") as timing_file:
            for index, sample in enumerate(samples):
                record, timing = run_sample(backend, sample, cfg, attempts_file)
                predictions[sample.photo_id] = record
                timings.append(timing)
                pred_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                pred_file.flush()
                timing_file.write(json.dumps(timing, allow_nan=False) + "\n")
                timing_file.flush()
                logger.metrics({"image_seconds": timing["seconds"], "attempts": timing["attempts"],
                                "image_failed": int(record["error"] is not None)}, step=index)
                if index < cfg.comet.log_images:
                    from .visualize import draw_prediction

                    logger.image(draw_prediction(open_sample(sample), record), str(sample.photo_id))
                log.info("%s/%s photo_id=%s boxes=%s attempts=%s seconds=%.3f", index + 1,
                         len(samples), sample.photo_id,
                         len(record["boxes"]) if record["boxes"] is not None else "ERROR",
                         timing["attempts"], timing["seconds"])
        processing_seconds = sum(t["seconds"] for t in timings)
        metrics = {"images": len(samples), "processing_seconds": processing_seconds,
                   "mean_image_seconds": processing_seconds / len(samples),
                   "images_per_second": len(samples) / processing_seconds,
                   "data_load_seconds": data_seconds, "model_load_seconds": load_seconds,
                   "wall_seconds": time.perf_counter() - started,
                   "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(cfg.model.device) / 2**20,
                   "device_peak_used_mib": trace.peak_mib,
                   "processing_errors": sum(p["error"] is not None for p in predictions.values()),
                   "parsing_errors": sum(t["parsing_errors"] for t in timings),
                   "validation_errors": sum(t["validation_errors"] for t in timings),
                   "runtime_errors": sum(t["runtime_errors"] for t in timings),
                   "total_attempts": sum(t["attempts"] for t in timings)}
        if cfg.mode == "test":
            truth = {s.photo_id: s.truth for s in samples}
            metrics.update(evaluate(predictions, truth, thresholds, cfg.evaluation.primary_iou))
        if metrics["processing_errors"] == 0:
            # Only successful records form a submission file. This performs no upload.
            (output_dir / "submission.jsonl").write_text((output_dir / "predictions.jsonl").read_text())
        write_json(output_dir / "metrics.json", metrics)
        logger.metrics(metrics)
        for name in ["config.yaml", "environment.json", "data.json", "model.json", "metrics.json",
                     "predictions.jsonl", "timings.jsonl", "gpu.jsonl"]:
            logger.asset(output_dir / name)
        if cfg.comet.log_raw_responses:
            logger.asset(output_dir / "attempts.jsonl")
        status = "complete" if metrics["processing_errors"] == 0 else "completed_with_errors"
        write_json(output_dir / "status.json", {"status": status})
        log.info("Results: %s", output_dir)
        return {"output_dir": str(output_dir), "metrics": metrics, "status": status}
    except BaseException as exc:
        write_json(output_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if backend is not None:
            backend.close()
        trace.close()
        if logger is not None:
            logger.close()
