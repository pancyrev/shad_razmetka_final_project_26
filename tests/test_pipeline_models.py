import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image

from butterfly_bench.data import load_samples
from butterfly_bench.metrics import evaluate
from scripts import pipeline_models as models
from scripts import prompt_search_common


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def new_inputs(tmp_path):
    photos = tmp_path / "photos.jsonl"
    gold = tmp_path / "gold.jsonl"
    originals = tmp_path / "originals"
    originals.mkdir()
    ids = [99881001, 99881002]
    photo_rows = []
    for i, pid in enumerate(ids):
        image = originals / f"{pid}.png"
        Image.new("RGB", (20, 30), color=(i, 100, 20)).save(image)
        photo_rows.append({"photo_id": pid, "photo_url": f"https://example.org/{pid}.png",
                           "image_path": str(image.relative_to(tmp_path))})
    write_rows(photos, photo_rows)
    write_rows(gold, [{"photo_id": pid, "boxes": []} for pid in reversed(ids)])
    return photos, gold, originals, tmp_path / "models"


def fake_cuda_run(cfg):
    output = Path(cfg.output_dir)
    samples, meta = load_samples(cfg.data, cfg.mode, cfg.seed)
    assert [s.photo_id for s in samples] == [99881001, 99881002]
    assert all(s.open_image().size == (20, 30) for s in samples)
    frozen = OmegaConf.load(models.ROOT / "results/final_400/frozen" / f"{cfg.model.name}.yaml")
    assert cfg.retries == frozen.retries and cfg.postprocess == frozen.postprocess
    assert cfg.model.revision == frozen.model.revision and cfg.model.prompt == frozen.model.prompt
    if cfg.model.name == "gemma":
        assert Path(cfg.model.server_binary) == models.ROOT / frozen.model.server_binary
    predictions = {s.photo_id: {"photo_id": s.photo_id, "boxes": [], "error": None} for s in samples}
    # A VLM that exhausts its retries is a legitimate completed benchmark outcome.
    if cfg.model.name == "gemma":
        predictions[99881002].update(boxes=None, error="ValidationError: invalid coordinates")
    metrics = evaluate(predictions, {s.photo_id: s.truth for s in samples})
    status = "completed_with_errors" if metrics["processing_errors"] else "complete"
    OmegaConf.save(cfg, output / "config.yaml")
    (output / "status.json").write_text(json.dumps({"status": status}))
    (output / "data.json").write_text(json.dumps(meta))
    (output / "metrics.json").write_text(json.dumps(metrics))
    write_rows(output / "predictions.jsonl", predictions.values())
    write_rows(output / "attempts.jsonl", [{"photo_id": s.photo_id} for s in samples])


def test_fresh_models_resume_only_matching_results(new_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_search_common, "GPU_LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(models, "run", fake_cuda_run)
    manifest = models.prepare_model_manifest(*new_inputs)
    results = {name: models.run_model(name, manifest, new_inputs[-1]) for name in models.MODEL_NAMES}
    assert results["gemma"]["status"] == "completed_with_errors"
    with monkeypatch.context() as patch:
        patch.setattr(models, "run", lambda _: pytest.fail("Should reuse verified result"))
        for name in models.MODEL_NAMES:
            assert models.run_model(name, manifest, new_inputs[-1]) == results[name]

    # Changing a prediction after completion forces a new attempt; the damaged file is preserved.
    old = Path(results["owlv2"]["output_dir"])
    (old / "predictions.jsonl").write_text("invalid")
    assert models.run_model("owlv2", manifest, new_inputs[-1])["output_dir"] != str(old)
    assert (old / "predictions.jsonl").read_text() == "invalid"
    labels = models.jsonl_rows(new_inputs[1])
    labels[0]["boxes"] = [{"left": 0, "top": 0, "width": .5, "height": .5}]
    write_rows(new_inputs[1], labels)
    changed = models.prepare_model_manifest(*new_inputs)
    assert changed != manifest and models.run_model("qwen", changed, new_inputs[-1])["metrics"]["fn"] == 1


def test_interrupted_run_and_image_integrity(new_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_search_common, "GPU_LOCK", tmp_path / "gpu.lock")
    manifest = models.prepare_model_manifest(*new_inputs)

    def stop(cfg):
        (Path(cfg.output_dir) / "status.json").write_text('{"status":"running"}')
        raise KeyboardInterrupt()

    monkeypatch.setattr(models, "run", stop)
    with pytest.raises(KeyboardInterrupt):
        models.run_model("gemma", manifest, new_inputs[-1])
    monkeypatch.setattr(models, "run", fake_cuda_run)
    assert models.run_model("gemma", manifest, new_inputs[-1])["metrics"]["images"] == 2
    assert len(list((new_inputs[-1] / "gemma").glob("attempt-*"))) == 2
    (new_inputs[2] / "99881001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Original changed"):
        models.run_model("gemma", manifest, new_inputs[-1])
    write_rows(new_inputs[1], [{"photo_id": 1, "boxes": []}])
    with pytest.raises(ValueError, match="IDs differ"):
        models.prepare_model_manifest(*new_inputs)


def test_manifest_preserves_explicit_historical_order(new_inputs):
    sample = [99881002, 99881001]
    manifest = models.prepare_model_manifest(*new_inputs, photo_ids=sample)
    assert [row["photo_id"] for row in models.jsonl_rows(manifest)] == sample
    for invalid in (sample[:-1], [sample[0], sample[0]], sample + [123], [123, sample[1]]):
        with pytest.raises(ValueError, match="exactly once"):
            models.prepare_model_manifest(*new_inputs, photo_ids=invalid)


def test_historical_parquet_identity_and_rgb_fallback(new_inputs, tmp_path, monkeypatch):
    manifest = models.prepare_model_manifest(*new_inputs)
    archive = tmp_path / "pinned.parquet"
    archive.write_bytes(b"mock historical parquet")
    rows = models.jsonl_rows(manifest)
    source_rows = [{"photo_id": row["photo_id"], "image": {"bytes": (manifest.parent / row["image_path"]).read_bytes()},
                    "boxes": row["boxes"]} for row in rows]
    info = {"repo_id": "owner/name", "revision": "a" * 40, "parquet_file": "test.parquet",
            "parquet_sha256": models.checksum(archive), "count": len(rows), "photo_ids": [r["photo_id"] for r in rows]}
    metadata = tmp_path / "dataset.json"
    metadata.write_text(json.dumps(info))

    class FakeParquet:
        def __init__(self, path):
            assert path == archive
            self.metadata = SimpleNamespace(num_rows=len(source_rows))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def iter_batches(self, batch_size, columns):
            assert batch_size < 10 and columns == ["photo_id", "image", "boxes"]
            yield SimpleNamespace(to_pylist=lambda: source_rows)

    def download(**kwargs):
        assert kwargs["revision"] == info["revision"] and kwargs["repo_id"] == info["repo_id"]
        assert kwargs["repo_type"] == "dataset" and Path(kwargs["cache_dir"]) == tmp_path
        return archive

    parquet = SimpleNamespace(ParquetFile=FakeParquet)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    # Lossless re-encoding changes bytes, but the pixel array actually seen by the model is identical.
    with Image.open(manifest.parent / rows[0]["image_path"]) as im:
        buf = BytesIO()
        im.save(buf, format="BMP")
    source_rows[0]["image"]["bytes"] = buf.getvalue()
    check = models.verify_historical_pixels(manifest, metadata, tmp_path)
    assert check["images"] == 2 and check["byte_identical"] == check["same_rgb_different_encoding"] == 1
    source_rows[0]["image"]["bytes"] = source_rows[1]["image"]["bytes"]
    with pytest.raises(ValueError, match="pixels differ"):
        models.verify_historical_pixels(manifest, metadata, tmp_path)
    archive.write_bytes(b"another parquet")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        models.verify_historical_pixels(manifest, metadata, tmp_path)
