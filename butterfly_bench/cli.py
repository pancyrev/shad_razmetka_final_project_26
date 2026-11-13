import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from .pipeline import run

    load_dotenv()
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = run(cfg)
    if result["metrics"]["processing_errors"]:
        raise SystemExit(2)
