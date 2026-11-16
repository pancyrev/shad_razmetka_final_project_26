import importlib
import json
from pathlib import Path

import pytest

from butterfly_bench.api import configure


def search_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prompt_search_common")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_current_evaluation_splits_are_rejected_for_prompt_tuning(monkeypatch, split):
    module = search_module(monkeypatch)
    cfg = configure([f"data.split={split}", f"data.revision={module.DATASET_REVISION}"])
    with pytest.raises(ValueError, match="100 train"):
        module.cached_tuning(cfg.data, "test", 42)
    with pytest.raises(ValueError, match="pinned train"):
        module.verify_training_config(cfg)


def test_prompt_search_run_uses_new_train_without_touching_archive(monkeypatch, tmp_path):
    from butterfly_bench import pipeline

    module = search_module(monkeypatch)
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    (tmp_path / "qwen").mkdir()
    (tmp_path / "qwen/prompts.json").write_text(json.dumps([{"name": "baseline", "prompt": "butterfly"}]))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args: None)
    observed = []

    def run(cfg):
        assert cfg.data.split == "train" and cfg.data.revision == module.DATASET_REVISION
        assert cfg.data.parquet_file == "data/train-00000-of-00001.parquet"
        assert pipeline.load_samples is module.cached_tuning
        assert "train-tuning" in cfg.comet.tags
        observed.append(cfg)
        return {"metrics": {"images": 100}}

    monkeypatch.setattr(pipeline, "run", run)
    module.run_one("qwen", 0, tmp_path / "run")
    assert len(observed) == 1
