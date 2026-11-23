"""Repeat the original selected full-record prompts on the fixed validation set."""

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
model = sys.argv[1]
if model not in {"qwen", "gemma"}:
    raise ValueError("Expected qwen or gemma")
output = Path(__file__).resolve().parent / model
os.environ["COMET_OFFLINE_DIRECTORY"] = str(output / "comet-offline")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

from prompt_search_common import cached_validation

from butterfly_bench import pipeline
from butterfly_bench.api import configure
from butterfly_bench.models import prompt_for

cfg = configure([f"model={model}_tuned",
                 "data.revision=1316e3c7147d4399a03f90684e191aae4d798f09"], output_dir=output)
best = json.loads((ROOT / "results/experiments/prompt_search_2026-11-15" / model /
                   "best_prompt.json").read_text())
assert cfg.model.prompt == best["prompt"]
assert prompt_for(cfg.model, 456).endswith("The photo_id for this image is 456.")
cfg.comet.experiment_name = f"restored-task-record-{model}-validation-100"
cfg.comet.tags = [*cfg.comet.tags, "restored-task-record", "validation", "retest"]
with patch.object(pipeline, "resolve_model", lambda config: config.revision), patch.object(
    pipeline, "load_samples", cached_validation
):
    result = pipeline.run(cfg)
assert result["metrics"]["images"] == 100
print(json.dumps(result))
