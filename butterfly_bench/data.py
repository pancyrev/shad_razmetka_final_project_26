import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from .schema import ValidationError, validate_boxes, validate_photo_id


@dataclass
class Sample:
    photo_id: int
    payload: dict
    truth: list | None = None

    def open_image(self):
        if "bytes" in self.payload:
            source = io.BytesIO(self.payload["bytes"])
        else:
            source = self.payload["path"]
        with Image.open(source) as image:
            # Preserve original pixel coordinates, as in the source annotations.
            # Do not silently apply EXIF rotation to only the input image.
            image.load()
            return image.convert("RGB")


def load_samples(cfg, mode, seed):
    rows = []
    metadata = {"source": cfg.source}
    if cfg.source == "hf":
        from datasets import Image as HFImage
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi().dataset_info(cfg.repo_id, revision=cfg.revision, token=False).sha
        path = hf_hub_download(cfg.repo_id, cfg.parquet_file, repo_type="dataset",
                               revision=revision, token=False)
        dataset = load_dataset("parquet", data_files={cfg.split: path}, split=cfg.split)
        if mode == "test" and "boxes" not in dataset.column_names:
            raise ValidationError(
                f"HF split {cfg.split!r} has no ground-truth boxes; mode=test requires labeled data. "
                "Select train/validation or use mode=inference for unlabeled images.")
        dataset = dataset.cast_column("image", HFImage(decode=False))
        metadata.update(repo_id=cfg.repo_id, revision=revision, split=cfg.split,
                        parquet_file=cfg.parquet_file)
        for row in dataset:
            image = row["image"]
            payload = {"bytes": image["bytes"]} if image.get("bytes") is not None else {"path": image["path"]}
            rows.append(Sample(row["photo_id"], payload, row.get("boxes") if mode == "test" else None))
    elif cfg.source == "manifest":
        path = Path(cfg.path).resolve()
        metadata.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for line in path.read_text().splitlines():
            if not line.strip():
                raise ValueError("blank lines are not allowed in input JSONL")
            row = json.loads(line)
            photo_id = validate_photo_id(row["photo_id"])
            local = row.get("image_path")
            if local:
                local = Path(local)
                local = local if local.is_absolute() else path.parent / local
            else:
                local = Path(cfg.image_root) / f"{photo_id}.jpg"
            payload = {"path": str(local), "photo_url": row.get("photo_url"),
                       "cache_dir": str(cfg.cache_dir), "request_timeout": cfg.request_timeout}
            rows.append(Sample(photo_id, payload, row.get("boxes") if mode == "test" else None))
    else:
        raise ValueError(f"unknown data source: {cfg.source}")
    ids = [validate_photo_id(row.photo_id) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate photo_id in input data")
    if cfg.photo_ids is not None:
        selected = set(cfg.photo_ids)
        if selected - set(ids):
            raise ValueError(f"unknown photo IDs: {sorted(selected - set(ids))}")
        rows = [row for row in rows if row.photo_id in selected]
    if cfg.shuffle:
        random.Random(seed).shuffle(rows)
    if cfg.limit is not None:
        if cfg.limit <= 0:
            raise ValueError("data.limit must be positive or null")
        rows = rows[:cfg.limit]
    if not rows:
        raise ValueError("input data selection is empty")
    if mode == "test":
        for row in rows:
            if row.truth is None:
                raise ValidationError(
                    f"photo_id={row.photo_id} has no ground-truth boxes; "
                    "mode=test requires labels (an empty array is a valid negative example).")
            row.truth = validate_boxes(row.truth)
    metadata["photo_ids"] = [row.photo_id for row in rows]
    metadata["count"] = len(rows)
    return rows, metadata


def open_sample(sample):
    if "bytes" not in sample.payload and not Path(sample.payload["path"]).exists():
        url = sample.payload.get("photo_url")
        if not url:
            raise FileNotFoundError(sample.payload["path"])
        cache = Path(sample.payload["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        # URL digest prevents sharing stale content across different manifests.
        target = cache / f"{sample.photo_id}_{hashlib.sha256(url.encode()).hexdigest()[:16]}.jpg"
        if not target.exists():
            response = requests.get(url, timeout=sample.payload["request_timeout"])
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image.verify()
            temporary = target.with_suffix(".part")
            temporary.write_bytes(response.content)
            temporary.replace(target)
        sample.payload["path"] = str(target)
    return sample.open_image()
