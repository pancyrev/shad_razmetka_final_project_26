import importlib.metadata
import json
import platform
import subprocess
import threading
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def environment_info():
    packages = {}
    for name in ["torch", "torchvision", "transformers", "datasets", "huggingface-hub",
                 "hydra-core", "comet-ml", "Pillow", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_commit": git.stdout.strip(),
            "git_dirty": bool(status.stdout.strip())}


class GPUTrace:
    """Sample device-wide utilization; torch peak allocation is recorded separately."""

    def __init__(self, output_dir, gpu_index=0):
        self.path = Path(output_dir) / "gpu.jsonl"
        self.gpu_index = gpu_index
        self.stop = threading.Event()
        self.thread = None
        self.peak_mib = 0

    def start(self):
        def loop():
            with self.path.open("w") as handle:
                while not self.stop.is_set():
                    try:
                        result = subprocess.run([
                            "nvidia-smi", f"--id={self.gpu_index}",
                            "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                            check=False)
                        if result.returncode == 0:
                            fields = [s.strip() for s in result.stdout.strip().split(",")]
                            record = dict(zip(["timestamp", "name", "used_mib", "total_mib", "gpu_percent"],
                                              fields, strict=True))
                            self.peak_mib = max(self.peak_mib, float(record["used_mib"]))
                            handle.write(json.dumps(record) + "\n")
                            handle.flush()
                    except (OSError, subprocess.TimeoutExpired, ValueError):
                        pass
                    self.stop.wait(1)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)


class CometLogger:
    def __init__(self, cfg, output_dir, config):
        import comet_ml

        self.experiment = None
        self.output_dir = Path(output_dir)
        mode = cfg.mode
        if mode not in {"auto", "online", "offline", "disabled"}:
            raise ValueError("comet.mode must be auto, online, offline, or disabled")
        self.info = {"requested_mode": mode, "mode": mode}
        if mode == "disabled":
            write_json(self.output_dir / "comet.json", self.info)
            return
        if mode == "auto":
            from comet_ml.config import get_config

            mode = "online" if get_config("comet.api_key") else "offline"
        kwargs = {"project_name": cfg.project_name, "auto_param_logging": False,
                  "auto_metric_logging": False, "log_code": False, "log_env_details": False,
                  "log_env_gpu": False, "log_env_cpu": False,
                  "log_env_disk": False, "log_env_network": False, "log_env_host": False,
                  "auto_log_co2": False, "log_graph": False,
                  "log_git_metadata": False, "log_git_patch": False,
                  "auto_output_logging": False}
        if cfg.workspace:
            kwargs["workspace"] = cfg.workspace
        if mode == "online":
            self.experiment = comet_ml.Experiment(**kwargs)
        else:
            self.experiment = comet_ml.OfflineExperiment(
                offline_directory=str(self.output_dir / "comet-offline"), **kwargs)
        self.info.update(mode=mode, experiment_key=self.experiment.get_key())
        if mode == "online":
            self.info["url"] = self.experiment.url
        self.experiment.set_name(cfg.experiment_name or self.output_dir.name)
        self.experiment.add_tags(list(cfg.tags))
        self.experiment.log_parameters(_flatten(config))
        write_json(self.output_dir / "comet.json", self.info)

    def metrics(self, values, step=None):
        if self.experiment:
            self.experiment.log_metrics(values, step=step)

    def asset(self, path):
        if self.experiment and Path(path).is_file():
            self.experiment.log_asset(str(path))

    def image(self, image, name):
        if self.experiment:
            self.experiment.log_image(image, name=name)

    def close(self):
        if self.experiment:
            self.experiment.end()


def _flatten(obj, prefix=""):
    result = {}
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        else:
            result[name] = json.dumps(value) if isinstance(value, list) else value
    return result
