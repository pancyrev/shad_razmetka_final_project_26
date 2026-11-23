"""Execute the updated demo on its default train subset, under the shared GPU lock."""

import json
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"]
notebook_path = ROOT / "notebooks/models/04_model_demo.ipynb"
nb = nbformat.read(notebook_path, as_version=4)
assert 'SPLIT = "train"' in nb.cells[3].source
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb, notebook_path)
outputs = "\n".join(output.get("text", "") for cell in nb.cells if cell.cell_type == "code"
                    for output in cell.outputs)
match = re.search(r"Результаты: (.+)", outputs)
assert match
directory = Path(match.group(1).strip()) / "grounding_dino"
from omegaconf import OmegaConf

cfg = OmegaConf.load(directory / "config.yaml")
assert cfg.data.split == "train"
assert cfg.data.revision == "863f76cfbcbcaf1563693b8a08e5128efa47a4c4"
metrics = json.loads((directory / "metrics.json").read_text())
assert metrics["images"] == 4 and metrics["processing_errors"] == 0
summary = {"notebook": str(notebook_path.relative_to(ROOT)), "split": cfg.data.split,
           "revision": cfg.data.revision, "images": 4,
           "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
           "processing_errors": 0, "output_dir": str(directory),
           "comet": json.loads((directory / "comet.json").read_text())}
(HERE / "notebook_check.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
