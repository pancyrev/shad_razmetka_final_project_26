import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
model = sys.argv[1]
output = ROOT / 'results/experiments/boxes_only_2026-11-16' / model
os.environ['COMET_OFFLINE_DIRECTORY'] = str(output / 'comet-offline')
from dotenv import load_dotenv

load_dotenv(ROOT / '.env')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpx2').setLevel(logging.WARNING)
from prompt_search_common import cached_validation

from butterfly_bench import pipeline
from butterfly_bench.api import configure

cfg = configure([f'model={model}_tuned',
                 'data.revision=1316e3c7147d4399a03f90684e191aae4d798f09'], output_dir=output)
cfg.comet.experiment_name = f'boxes-only-{model}-validation-100'
cfg.comet.tags = [*cfg.comet.tags, 'boxes-only', 'validation', 'protocol-change']
assert cfg.model.response_format == 'boxes_only'
assert 'photo_id' not in cfg.model.prompt and '"error"' not in cfg.model.prompt
with patch.object(pipeline, 'resolve_model', lambda config: config.revision), patch.object(
    pipeline, 'load_samples', cached_validation
):
    result = pipeline.run(cfg)
assert result['metrics']['images'] == 100
print(result)
