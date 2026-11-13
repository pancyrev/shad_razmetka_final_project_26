import logging

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from .pipeline import run

    load_dotenv()
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = run(cfg)
    if result["metrics"]["processing_errors"]:
        if HydraConfig.get().mode == RunMode.MULTIRUN:
            logging.getLogger(__name__).error(
                "Run has processing errors; results: %s. Continuing remaining models.", result["output_dir"])
        else:
            raise SystemExit(2)
    return result
