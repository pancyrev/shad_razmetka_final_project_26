"""The same Hydra configuration and pipeline for notebooks and the CLI."""

from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir


def configure(overrides=None, output_dir=None):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).parent / "conf")):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    cfg.output_dir = str(Path(output_dir or (
        f"outputs/notebook/{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_{cfg.model.name}_{cfg.mode}")).resolve())
    return cfg

