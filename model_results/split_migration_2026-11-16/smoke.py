"""Check four fixed CUDA backends on train/validation only; no contest test."""

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from omegaconf import OmegaConf

from butterfly_bench import pipeline
from butterfly_bench.api import configure

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
model, split = sys.argv[1:]
assert model in {"owlv2", "grounding_dino", "qwen", "gemma"}
assert split in {"train", "validation", "inference_train"}
inference = split == "inference_train"
data_split = "train" if inference else split
info = json.loads((HERE / f"{data_split}.json").read_text())
old = OmegaConf.load(ROOT / "model_results/test_2026-11-16/frozen" / f"{model}.yaml")
output = HERE / split / model
os.environ["COMET_OFFLINE_DIRECTORY"] = str(output / "comet-offline")
preset = f"{model}_tuned" if model in {"qwen", "gemma"} else model
cfg = configure([f"model={preset}", f"mode={'inference' if inference else 'test'}",
                 f"data.split={data_split}", f"data.revision={info['revision']}",
                 f"model.revision={old.model.revision}"], output_dir=output)
cfg.data.photo_ids = info["smoke_ids"]
cfg.postprocess = old.postprocess
assert cfg.model == old.model
cfg.comet.experiment_name = f"split-migration-{split}-{model}-4"
cfg.comet.tags = [*cfg.comet.tags, "split-migration", split, "smoke"]
result = pipeline.run(cfg)
assert result["metrics"]["images"] == 4
assert result["metrics"]["runtime_errors"] == 0
assert result["metrics"]["processing_errors"] == 0
if inference:
    assert "f1" not in result["metrics"] and "precision" not in result["metrics"]
else:
    source = ("model_results/validation_2026-11-13/owlv2_nms_0.5" if model == "owlv2" else
              "model_results/validation_2026-11-13/grounding_dino" if model == "grounding_dino" else
              "model_results/restored_task_record_2026-11-16/" + model) if split == "train" else (
                  "model_results/test_2026-11-16/" + model)
    previous = {r["photo_id"]: r for r in map(json.loads, (ROOT / source / "predictions.jsonl").read_text().splitlines())}
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert all(record == previous[record["photo_id"]] for record in records)
    (output / "regression.json").write_text(json.dumps({
        "compared_to": source, "all_four_predictions_identical": True}) + "\n")
print(json.dumps(result), flush=True)
