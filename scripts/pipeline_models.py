"""Evaluate the four frozen model configurations on the photos from one new crowd run."""

import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from butterfly_bench.data import load_samples
from butterfly_bench.metrics import evaluate
from butterfly_bench.pipeline import run
from butterfly_bench.schema import validate_boxes, validate_photo_id
from scripts.download_selected import checksum, image_path, jsonl_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("owlv2", "grounding_dino", "qwen", "gemma")


def _by_id(path):
    rows = jsonl_rows(Path(path))
    indexed = {validate_photo_id(row["photo_id"]): row for row in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError(f"Expected nonempty rows with unique photo_id in {path}")
    return indexed


def _atomic_write(path, contents):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(contents)
    os.replace(tmp.name, path)


def prepare_model_manifest(main_photos_path, final_annotations_path, image_dir, results_dir, *, photo_ids=None):
    """Join the actual sample and reviewed labels by ID, retaining each original file's hash."""
    main_path = Path(main_photos_path).resolve()
    photos, gold = _by_id(main_path), _by_id(final_annotations_path)
    if photos.keys() != gold.keys():
        raise ValueError("Main photo IDs and final annotation IDs differ")
    ordered = list(photos) if photo_ids is None else [validate_photo_id(pid) for pid in photo_ids]
    if len(ordered) != len(photos) or set(ordered) != photos.keys():
        raise ValueError("Ordered photo_ids must contain every main photo exactly once")
    directory = Path(results_dir).resolve() / "datasets"
    directory.mkdir(exist_ok=True, parents=True)
    rows = []
    for pid in ordered:
        photo = photos[pid]
        label = gold[pid]
        if label.get("photo_url", photo["photo_url"]) != photo["photo_url"]:
            raise ValueError(f"Image URL changed for photo {pid}")
        if photo.get("image_path"):
            local = (main_path.parent / photo["image_path"]).resolve(strict=True)
        else:
            local = image_path(photo, Path(image_dir)).resolve(strict=True)
        digest = checksum(local)
        if photo.get("sha256", digest) != digest:
            raise ValueError(f"Image checksum changed for photo {pid}")
        rows.append({"photo_id": pid, "photo_url": photo["photo_url"],
                     "image_path": os.path.relpath(local, directory), "sha256": digest,
                     "boxes": validate_boxes(label["boxes"])})
    contents = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    path = directory / (hashlib.sha256(contents.encode()).hexdigest() + ".jsonl")
    _atomic_write(path, contents)
    return path


def verify_historical_pixels(manifest, dataset_info_path, cache_dir):
    """Compare the rebuilt inputs with the exact Parquet used in the original experiment."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    manifest = Path(manifest).resolve()
    rows = _by_id(manifest)
    dataset = json.loads(Path(dataset_info_path).read_text())
    if list(rows) != dataset["photo_ids"] or len(rows) != dataset["count"]:
        raise ValueError("Manifest photo IDs or order differ from the original model run")
    parquet = Path(hf_hub_download(repo_id=dataset["repo_id"], filename=dataset["parquet_file"],
                                  repo_type="dataset", revision=dataset["revision"], token=False,
                                  cache_dir=str(Path(cache_dir).resolve())))
    if checksum(parquet) != dataset["parquet_sha256"]:
        raise ValueError(f"SHA256 mismatch for historical Parquet: {parquet}")
    byte_identical = 0
    index = 0
    with pq.ParquetFile(parquet) as source:
        # Read the embedded image bytes without retaining the full ~500 MB table in memory.
        if source.metadata.num_rows != len(rows):
            raise ValueError("Row count differs from the original model run")
        for batch in source.iter_batches(batch_size=4, columns=["photo_id", "image", "boxes"]):
            for original in batch.to_pylist():
                pid = original["photo_id"]
                if pid != dataset["photo_ids"][index]:
                    raise ValueError(f"Historical Parquet photo order differs at row {index}")
                current = rows[pid]
                if validate_boxes(original["boxes"]) != current["boxes"]:
                    raise ValueError(f"Rebuilt ground truth differs from the historical boxes for {pid}")
                local = manifest.parent / current["image_path"]
                if checksum(local) != current["sha256"]:
                    raise ValueError(f"Original changed after annotation: {local}")
                encoded = original["image"]["bytes"]
                if not encoded:
                    raise ValueError(f"Historical Parquet lacks embedded image bytes for {pid}")
                if hashlib.sha256(encoded).hexdigest() == current["sha256"]:
                    byte_identical += 1
                else:
                    # Match Sample.open_image: original orientation, RGB conversion, no EXIF transpose.
                    with Image.open(BytesIO(encoded)) as historic, Image.open(local) as rebuilt:
                        if (historic.size != rebuilt.size
                                or historic.convert("RGB").tobytes() != rebuilt.convert("RGB").tobytes()):
                            raise ValueError(f"Rebuilt pixels differ from the historical model input for {pid}")
                index += 1
    return {"images": index, "byte_identical": byte_identical, "same_rgb_different_encoding": index - byte_identical,
            "parquet_sha256": dataset["parquet_sha256"]}


def _config(name, manifest):
    cfg = OmegaConf.load(ROOT / "results/final_400/frozen" / f"{name}.yaml")
    cfg.data = OmegaConf.load(ROOT / "butterfly_bench/conf/data/manifest.yaml")
    cfg.data.path = str(manifest)
    cfg.data.image_root = str(manifest.parent)
    cfg.data.cache_dir = str(ROOT.parent / "downloads/final-project/manifest-cache")
    cfg.data.revision = None
    cfg.comet.mode = "disabled"
    cfg.comet.experiment_name = f"fresh-{name}"
    cfg.comet.tags = ["butterfly-detection", "fresh-crowd"]
    if name == "gemma":
        cfg.model.server_binary = str(ROOT / cfg.model.server_binary)
    return cfg


def _read_result(name, output, cfg, samples):
    metrics = json.loads((output / "metrics.json").read_text())
    status = json.loads((output / "status.json").read_text())["status"]
    if status not in {"complete", "completed_with_errors"}:
        raise ValueError(f"Incomplete model run in {output}")
    # Check order/coverage as well as TP/FP/FN, including exhausted retries.
    predictions = _by_id(output / "predictions.jsonl")
    truth = {sample.photo_id: sample.truth for sample in samples}
    saved_data = json.loads((output / "data.json").read_text())
    if (list(predictions) != list(truth) or saved_data["photo_ids"] != list(truth)
            or saved_data["sha256"] != checksum(Path(cfg.data.path))):
        raise ValueError(f"Model run used different photos or labels: {output}")
    checked = evaluate(predictions, truth, cfg.evaluation.iou_thresholds, cfg.evaluation.primary_iou)
    if (any(value != metrics.get(key) for key, value in checked.items())
            or (status == "complete") != (checked["processing_errors"] == 0)):
        raise ValueError(f"Metrics or status disagree with predictions: {output}")
    cfg.output_dir = str(output)
    if OmegaConf.load(output / "config.yaml") != cfg:
        raise ValueError(f"Model configuration changed: {output}")
    return {"model": name, "status": status, "output_dir": str(output), "metrics": metrics}


def run_model(name, manifest, results_dir):
    """Execute one real CUDA model or resume its verified result in the same run directory."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}")
    manifest = Path(manifest).resolve(strict=True)
    cfg = _config(name, manifest)
    samples, _ = load_samples(cfg.data, cfg.mode, cfg.seed)
    for row in jsonl_rows(manifest):
        image = manifest.parent / row["image_path"]
        if checksum(image) != row["sha256"]:
            raise ValueError(f"Original changed after annotation: {image}")

    # Changing the reviewed data, frozen parameters or implementation starts a separate attempt.
    sources = [Path(__file__).resolve(), *sorted((ROOT / "butterfly_bench").glob("*.py"))]
    inputs = {"manifest": checksum(manifest),
              "frozen": checksum(ROOT / "results/final_400/frozen" / f"{name}.yaml"),
              "effective_config": hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest(),
              "sources": {p.name: checksum(p) for p in sources}}
    target = Path(results_dir).resolve() / name
    target.mkdir(parents=True, exist_ok=True)
    receipt = target / "completed.json"
    os.environ.setdefault("HF_HOME", str(ROOT.parent / "downloads/final-project/huggingface"))

    # Use the same shared device lock as the archived model experiments.
    import fcntl

    from scripts.prompt_search_common import GPU_LOCK

    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            previous = target / saved["attempt"]
            if (saved["inputs"] == inputs and previous.parent == target
                    and all((previous / f).is_file() and checksum(previous / f) == digest
                            for f, digest in saved["artifacts"].items())):
                return _read_result(name, previous, cfg, samples)

        output = Path(tempfile.mkdtemp(prefix="attempt-", dir=target))
        cfg.output_dir = str(output)
        run(cfg)
        result = _read_result(name, output, cfg, samples)
        artifacts = {f: checksum(output / f) for f in (
            "config.yaml", "data.json", "status.json", "metrics.json", "attempts.jsonl", "predictions.jsonl"
        )}
        _atomic_write(receipt, json.dumps({"attempt": output.name, "inputs": inputs, "artifacts": artifacts},
                                         indent=2) + "\n")
        return result
