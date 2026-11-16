"""Check only renamed labeled splits against pinned historical Parquet files."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from butterfly_bench.api import configure
from butterfly_bench.data import load_samples, open_sample

REVISION = "863f76cfbcbcaf1563693b8a08e5128efa47a4c4"
OLD_REVISION = "1316e3c7147d4399a03f90684e191aae4d798f09"
rows = []
for split, old_split, expected_count in (("train", "validation", 100), ("validation", "test", 400)):
    cfg = configure([f"data.split={split}", f"data.revision={REVISION}"])
    samples, metadata = load_samples(cfg.data, "test", cfg.seed)
    assert len(samples) == expected_count
    # Both old files were cached during previous campaigns. Never access current test.
    old_path = hf_hub_download(cfg.data.repo_id, f"data/{old_split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=OLD_REVISION, token=False,
                               local_files_only=True)
    original = pq.read_table(old_path).to_pylist()
    for sample, previous in zip(samples, original, strict=True):
        assert sample.photo_id == previous["photo_id"]
        assert sample.truth == previous["boxes"]
        assert sample.payload["bytes"] == previous["image"]["bytes"]
        open_sample(sample).verify()
    current_path = Path(hf_hub_download(cfg.data.repo_id, cfg.data.parquet_file,
                                       repo_type="dataset", revision=REVISION, token=False,
                                       local_files_only=True))
    ids = metadata["photo_ids"]
    # Deterministic smoke subset: a negative, a multi-object example if present, then others.
    chosen = []
    for predicate in (lambda sample: not sample.truth, lambda sample: len(sample.truth) > 1,
                      lambda sample: bool(sample.truth), lambda sample: True):
        for sample in samples:
            if predicate(sample) and sample.photo_id not in chosen:
                chosen.append(sample.photo_id)
                break
    assert len(chosen) == 4
    smoke_ids = [photo_id for photo_id in ids if photo_id in chosen]
    metadata.update(expected_count=expected_count, previous_split=old_split,
                    previous_revision=OLD_REVISION, labels_and_images_identical=True,
                    all_images_decoded=True, ground_truth_boxes=sum(len(s.truth) for s in samples),
                    negative_images=sum(not s.truth for s in samples), smoke_ids=smoke_ids,
                    parquet_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest())
    (HERE / f"{split}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows.append(metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "photo_ids"}), flush=True)
assert not set(rows[0]["photo_ids"]) & set(rows[1]["photo_ids"])
(HERE / "dataset_check.json").write_text(json.dumps({
    "revision": REVISION, "renamed_labeled_splits_verified": True,
    "train_validation_disjoint": True, "current_test_downloaded_or_used": False,
    "splits": [{key: row[key] for key in ("split", "count", "ground_truth_boxes", "negative_images")}
               for row in rows]}, indent=2) + "\n")
