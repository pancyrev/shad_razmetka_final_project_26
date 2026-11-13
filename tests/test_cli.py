from types import SimpleNamespace

import pytest
from hydra.types import RunMode
from omegaconf import OmegaConf

from butterfly_bench import cli


@pytest.mark.parametrize("mode", [RunMode.RUN, RunMode.MULTIRUN])
def test_failed_images_do_not_cancel_remaining_multirun_models(monkeypatch, mode):
    result = {"output_dir": "outputs/example", "metrics": {"processing_errors": 1}}
    monkeypatch.setattr("butterfly_bench.pipeline.run", lambda cfg: result)
    monkeypatch.setattr(cli.HydraConfig, "get", lambda: SimpleNamespace(mode=mode))
    if mode == RunMode.RUN:
        with pytest.raises(SystemExit) as exc:
            cli.main.__wrapped__(OmegaConf.create({}))
        assert exc.value.code == 2
    else:
        assert cli.main.__wrapped__(OmegaConf.create({})) == result
