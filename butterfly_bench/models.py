import base64
import io
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .schema import detector_record


@dataclass
class ModelOutput:
    text: str | None = None
    record: dict | None = None
    scores: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


def resolve_model(cfg):
    from huggingface_hub import HfApi

    return HfApi().model_info(cfg.repo_id, revision=cfg.revision, token=False).sha


def torch_dtype(name):
    import torch

    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported unquantized dtype: {name}")
    return getattr(torch, name)


def prompt_for(cfg, photo_id, feedback=None):
    prompt = f"{cfg.prompt}\nThe photo_id for this image is {photo_id}."
    if feedback:
        prompt += ("\nYour previous response was rejected: " + feedback[:1200]
                   + ". Inspect this image again and return a corrected single JSONL record.")
    return prompt


class TransformersDetector:
    def __init__(self, cfg, revision, **_):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.repo_id, revision=revision, token=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg.repo_id, revision=revision, token=False, dtype=torch_dtype(cfg.dtype))
        self.model.to(cfg.device).eval()
        self.metadata = {"revision": revision, "dtype": str(self.model.dtype),
                         "device": str(self.model.device), "quantized": False}

    def predict(self, image, photo_id, *, attempt, seed, feedback=None):
        import torch

        prompt = self.cfg.prompt
        # Grounding DINO separates category phrases with periods.
        if self.cfg.name == "grounding_dino" and not prompt.rstrip().endswith("."):
            prompt = prompt.rstrip() + "."
        text = [[prompt]] if self.cfg.name == "owlv2" else prompt
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = inputs.to(self.cfg.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.cfg.device)
        if self.cfg.name == "owlv2":
            result = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.cfg.threshold, target_sizes=target_sizes)[0]
        else:
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.cfg.threshold,
                text_threshold=self.cfg.text_threshold, target_sizes=target_sizes)[0]
        coords = result["boxes"].float().cpu().tolist()
        scores = result["scores"].float().cpu().tolist()
        record, clipped = detector_record(photo_id, coords, scores, image.size,
                                          clip=self.cfg.clip_boxes)
        return ModelOutput(record=record, scores=scores,
                           details={"raw_boxes_xyxy": coords, "scores": scores,
                                    "clipped_boxes": clipped, "effective_prompt": prompt})

    def close(self):
        del self.model
        import torch

        torch.cuda.empty_cache()


class TransformersVLM:
    def __init__(self, cfg, revision, **_):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(
            cfg.repo_id, revision=revision, token=False,
            size={"shortest_edge": cfg.min_pixels, "longest_edge": cfg.max_pixels})
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.repo_id, revision=revision, token=False, dtype=torch_dtype(cfg.dtype),
            attn_implementation="sdpa").to(cfg.device).eval()
        self.metadata = {"revision": revision, "dtype": str(self.model.dtype),
                         "device": str(self.model.device), "quantized": False,
                         "max_pixels": cfg.max_pixels, "min_pixels": cfg.min_pixels}

    def predict(self, image, photo_id, *, attempt, seed, feedback=None):
        import torch
        from transformers import set_seed

        set_seed(seed)
        prompt = prompt_for(self.cfg, photo_id, feedback)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.cfg.device)
        temperature = self.cfg.temperature if attempt == 1 else self.cfg.retry_temperature
        generation = {"max_new_tokens": self.cfg.max_new_tokens,
                      "do_sample": temperature > 0, "use_cache": True}
        if temperature > 0:
            generation["temperature"] = temperature
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, **generation)
        generated = tokens[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        return ModelOutput(text=text, details={"output_tokens": generated.shape[1],
                                              "effective_prompt": prompt})

    def close(self):
        del self.model
        import torch

        torch.cuda.empty_cache()


class LlamaCppVLM:
    """Own a CUDA llama-server process or connect to an explicitly configured one."""

    def __init__(self, cfg, revision, output_dir, **_):
        from huggingface_hub import hf_hub_download

        self.cfg = cfg
        self.process = None
        self.log_handle = None
        self.metadata = {"revision": revision, "quantization": cfg.dtype,
                         "gguf_file": cfg.gguf_file, "mmproj_file": cfg.mmproj_file}
        if cfg.server_url:
            self.url = cfg.server_url.rstrip("/")
            # External server's actual weights/GPU placement are operator-controlled.
            self.metadata["external_server"] = True
            self._check_health()
            return
        binary = Path(cfg.server_binary).resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"{binary}: build CUDA llama.cpp with scripts/setup_llama_cpp.sh")
        model = hf_hub_download(cfg.repo_id, cfg.gguf_file, revision=revision, token=False)
        mmproj = hf_hub_download(cfg.repo_id, cfg.mmproj_file, revision=revision, token=False)
        port = cfg.port
        if port == 0:
            with socket.socket() as sock:
                sock.bind((cfg.host, 0))
                port = sock.getsockname()[1]
        self.url = f"http://{cfg.host}:{port}"
        command = [str(binary), "--model", model, "--mmproj", mmproj,
                   "--host", cfg.host, "--port", str(port),
                   "--n-gpu-layers", str(cfg.gpu_layers), "--main-gpu", str(cfg.main_gpu),
                   "--ctx-size", str(cfg.context_size), "--batch-size", str(cfg.batch_size),
                   "--ubatch-size", str(cfg.ubatch_size), "--threads", str(cfg.threads),
                   "--image-max-tokens", str(cfg.image_max_tokens), "--verbosity", "4",
                   "--parallel", "1", "--flash-attn", "on", "--no-warmup",
                   "--fit", "off", "--jinja", *map(str, cfg.extra_args)]
        self.metadata["server_command"] = command
        version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30,
                                 check=False)
        self.metadata["llama_cpp_version"] = (version.stdout + version.stderr).strip()
        self.log_handle = (Path(output_dir) / "llama-server.log").open("w")
        self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                        env=os.environ.copy())
        try:
            deadline = time.monotonic() + cfg.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("llama-server exited; see llama-server.log")
                try:
                    self._check_health()
                    break
                except (requests.RequestException, ValueError):
                    time.sleep(0.5)
            else:
                raise TimeoutError("llama-server startup timed out; see llama-server.log")
            # A CPU-only build must never masquerade as the requested CUDA run.
            log_text = (Path(output_dir) / "llama-server.log").read_text()
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log_text)
            if ("CUDA" not in log_text or not offload or int(offload[1]) == 0
                    or offload[1] != offload[2]):
                raise RuntimeError("Full CUDA layer offload was not confirmed; see llama-server.log")
            self.metadata["gpu_layers_offloaded"] = int(offload[1])
        except BaseException:
            self.close()
            raise

    def _check_health(self):
        response = requests.get(f"{self.url}/health", timeout=5)
        response.raise_for_status()
        if response.json().get("status") != "ok":
            raise ValueError("llama-server is not healthy")

    def predict(self, image, photo_id, *, attempt, seed, feedback=None):
        prompt = prompt_for(self.cfg, photo_id, feedback)
        processed = image.copy()
        processed.thumbnail((self.cfg.image_max_side, self.cfg.image_max_side))
        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        temperature = self.cfg.temperature if attempt == 1 else self.cfg.retry_temperature
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt}]}],
            "max_tokens": self.cfg.max_new_tokens, "temperature": temperature,
            "seed": seed, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
        response = requests.post(f"{self.url}/v1/chat/completions", json=payload,
                                 timeout=self.cfg.request_timeout)
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"].get("content")
        return ModelOutput(text=text, details={"usage": body.get("usage"),
                                              "timings": body.get("timings"),
                                              "finish_reason": body["choices"][0].get("finish_reason"),
                                              "effective_prompt": prompt,
                                              "input_size": list(processed.size)})

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def create_backend(cfg, revision, output_dir):
    classes = {"transformers_detector": TransformersDetector,
               "transformers_vlm": TransformersVLM, "llama_cpp": LlamaCppVLM}
    if cfg.backend not in classes:
        raise ValueError(f"unknown model backend: {cfg.backend}")
    return classes[cfg.backend](cfg, revision=revision, output_dir=output_dir)
