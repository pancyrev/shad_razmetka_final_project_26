import io
import json

import pytest
from PIL import Image

from butterfly_bench.api import configure
from butterfly_bench.data import Sample
from butterfly_bench.models import ModelOutput, prompt_for
from butterfly_bench.pipeline import run_sample


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def predict(self, image, photo_id, **kwargs):
        self.calls.append(kwargs)
        return ModelOutput(text=next(self.outputs))


def sample():
    buf = io.BytesIO()
    Image.new("RGB", (32, 24)).save(buf, format="PNG")
    return Sample(7, {"bytes": buf.getvalue()}, [])


def test_parser_and_schema_failures_both_trigger_feedback_and_retries():
    backend = FakeModel(['bad json', '{"boxes":[{"left":0.9,"top":0,"width":0.5,"height":0.2}]}',
                         '{"boxes":[]}'])
    handle = io.StringIO()
    result, timing = run_sample(backend, sample(), configure(["model=qwen"]), handle)
    assert result == {"photo_id": 7, "boxes": [], "error": None}
    assert timing["attempts"] == 3
    assert timing["parsing_errors"] == timing["validation_errors"] == 1
    assert backend.calls[0]["feedback"] is None
    assert "ParseError" in backend.calls[1]["feedback"]
    assert "ValidationError" in backend.calls[2]["feedback"]
    assert len({call["seed"] for call in backend.calls}) == 3
    assert [json.loads(line)["stage"] for line in handle.getvalue().splitlines()] == [
        "parsing", "validation", "success"]


def test_exhausted_retries_do_not_become_successful_empty_detection():
    result, timing = run_sample(FakeModel(['bad'] * 3), sample(), configure(), io.StringIO())
    assert result["boxes"] is None
    assert result["error"] is not None
    assert timing["parsing_errors"] == 3


def test_models_start_with_identical_vlm_prompts():
    assert configure(["model=qwen"]).model.prompt == configure(["model=gemma"]).model.prompt


@pytest.mark.parametrize("model", ["qwen", "gemma", "qwen_tuned", "gemma_tuned"])
def test_current_prompts_never_receive_image_id_or_request_metadata(model):
    cfg = configure([f"model={model}"]).model
    assert cfg.response_format == "boxes_only"
    for feedback in (None, "ValidationError: box 0: invalid origin or nonpositive size"):
        prompt = prompt_for(cfg, 9876543210, feedback)
        assert "9876543210" not in prompt
        assert "photo_id" not in prompt
        assert '"error"' not in prompt
        if feedback:
            assert feedback in prompt


def test_archived_prompts_can_still_be_reproduced_explicitly():
    cfg = configure(["model=qwen"]).model
    cfg.response_format = "task_record"
    assert prompt_for(cfg, 7).endswith("The photo_id for this image is 7.")


def test_unreadable_image_is_a_runtime_error_before_model_call(tmp_path):
    broken = Sample(7, {"path": str(tmp_path / "absent.png")})
    backend = FakeModel([])
    handle = io.StringIO()
    result, timing = run_sample(backend, broken, configure(), handle)
    assert result["boxes"] is None
    assert "FileNotFoundError" in result["error"]
    assert timing["runtime_errors"] == 3
    assert backend.calls == []


def test_structured_detector_result_still_requires_schema_validation():
    class Detector:
        def predict(self, image, photo_id, **kwargs):
            return ModelOutput(record={"photo_id": photo_id, "boxes": [
                {"left": 0.9, "top": 0, "width": 0.5, "height": 0.2}], "error": None}, scores=[0.8])

    result, timing = run_sample(Detector(), sample(), configure(), io.StringIO())
    assert result["boxes"] is None
    assert timing["parsing_errors"] == 0
    assert timing["validation_errors"] == 3
