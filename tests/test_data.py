import json
import sys
from types import SimpleNamespace

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


def fake_hf(monkeypatch, rows):
    """Exercise the HF loader without downloading any split or image."""
    requests = []

    class Dataset:
        @property
        def column_names(self):
            return list(rows[0])

        def cast_column(self, *args):
            return self

        def __iter__(self):
            return iter(rows)

    def download(repo_id, filename, **kwargs):
        requests.append(filename)
        return filename

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        Image=lambda **kwargs: None, load_dataset=lambda *args, **kwargs: Dataset()))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="pinned")),
        hf_hub_download=download))
    return requests


@pytest.mark.parametrize("split", ["train", "validation"])
def test_renamed_hf_labeled_splits_and_empty_negative_labels(monkeypatch, split):
    requests = fake_hf(monkeypatch, [
        {"photo_id": 1, "image": {"bytes": b"image"}, "boxes": []},
        {"photo_id": 2, "image": {"bytes": b"image"},
         "boxes": [{"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}]},
    ])
    cfg = configure([f"data.split={split}"])
    samples, metadata = load_samples(cfg.data, "test", 42)
    assert requests == [f"data/{split}-00000-of-00001.parquet"]
    assert metadata["split"] == split and metadata["revision"] == "pinned"
    assert samples[0].truth == [] and len(samples[1].truth) == 1


def test_hf_without_labels_requires_inference(monkeypatch):
    fake_hf(monkeypatch, [{"photo_id": 1, "image": {"bytes": b"image"}}])
    cfg = configure(["data.split=unlabeled"])
    with pytest.raises(ValidationError, match="no ground-truth boxes"):
        load_samples(cfg.data, "test", 42)
    samples, _ = load_samples(cfg.data, "inference", 42)
    assert samples[0].truth is None


def test_hf_null_labels_are_not_treated_as_negative_examples(monkeypatch):
    fake_hf(monkeypatch, [{"photo_id": 1, "image": {"bytes": b"image"}, "boxes": None}])
    cfg = configure(["data.split=validation"])
    with pytest.raises(ValidationError, match="photo_id=1 has no ground-truth"):
        load_samples(cfg.data, "test", 42)
