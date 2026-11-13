import json

import pytest
from PIL import Image

from butterfly_bench.api import configure
from butterfly_bench.data import load_samples, open_sample
from butterfly_bench.schema import ValidationError


def test_manifest_inference_ignores_labels_and_uses_relative_image_path(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (40, 20)).save(image_path)
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({"photo_id": 1, "image_path": "image.png", "boxes": "invalid"}) + "\n")
    cfg = configure(["data=manifest"])
    cfg.data.path = str(path)
    samples, metadata = load_samples(cfg.data, "inference", 42)
    assert samples[0].truth is None
    assert open_sample(samples[0]).size == (40, 20)
    assert metadata["photo_ids"] == [1]
    with pytest.raises(ValidationError):
        load_samples(cfg.data, "test", 42)


def test_duplicate_ids_and_missing_ground_truth_are_rejected(tmp_path):
    path = tmp_path / "dataset.jsonl"
    row = json.dumps({"photo_id": 1}) + "\n"
    path.write_text(row * 2)
    cfg = configure(["data=manifest"])
    cfg.data.path = str(path)
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(cfg.data, "inference", 42)
    path.write_text(row)
    with pytest.raises(ValidationError):
        load_samples(cfg.data, "test", 42)


def test_same_seed_selects_same_ids_for_all_models(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps({"photo_id": i, "boxes": []}) + "\n" for i in range(20)))
    selections = []
    for model in ["owlv2", "grounding_dino", "qwen", "gemma"]:
        cfg = configure([f"model={model}", "data=manifest", "data.shuffle=true", "data.limit=5"])
        cfg.data.path = str(path)
        _, metadata = load_samples(cfg.data, "test", cfg.seed)
        selections.append(metadata["photo_ids"])
    assert all(ids == selections[0] for ids in selections)
