import json

import pytest

from butterfly_bench.schema import (
    ParseError,
    ValidationError,
    detector_record,
    parse_jsonl_response,
    validate_record,
)


def record(boxes=None):
    return {"photo_id": 7, "boxes": boxes or [], "error": None}


def test_successful_empty_and_nonempty_answers():
    assert parse_jsonl_response('{"boxes":[]}', 7) == record()
    assert parse_jsonl_response('{"boxes":[]}', 42) == {"photo_id": 42, "boxes": [], "error": None}
    box = {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
    assert parse_jsonl_response(json.dumps({"boxes": [box]}), 7) == record([box])


@pytest.mark.parametrize("text", [
    '```json\n{"boxes":[]}\n```',
    '{"boxes":[]}\n{"boxes":[]}',
    '{"boxes":[],"boxes":[]}',
    '{"boxes":[NaN]}',
    'answer: {"boxes":[]}',
    '',
])
def test_bad_jsonl_is_not_silently_repaired(text):
    with pytest.raises(ParseError):
        parse_jsonl_response(text, 7)


@pytest.mark.parametrize("obj", [
    {}, [],
    {"boxes": None},
    {"photo_id": 8, "boxes": []},
    {"boxes": [], "error": None},
    {"boxes": [], "confidence": 0.9},
    record(),
    {"boxes": [{"left": 0.9, "top": 0, "width": 0.2, "height": 0.1}]},
    {"boxes": [{"left": 0, "top": 0, "width": 0, "height": 0.1}]},
    {"boxes": [{"left": 0, "top": 0, "width": True, "height": 0.1}]},
    {"boxes": [{"left": 100, "top": 20, "width": 30, "height": 40}]},
    {"boxes": [{"left": 0, "top": 0, "width": "0.1", "height": 0.1}]},
])
def test_schema_rejects_impossible_values(obj):
    with pytest.raises(ValidationError):
        parse_jsonl_response(json.dumps(obj), 7)


@pytest.mark.parametrize("obj", [
    {"photo_id": 8, "boxes": [], "error": None},
    {"photo_id": True, "boxes": [], "error": None},
    {"photo_id": 7, "boxes": [], "error": "failed"},
    {"photo_id": 7, "boxes": []},
])
def test_full_record_validation_still_checks_trusted_metadata(obj):
    with pytest.raises(ValidationError):
        validate_record(obj, 7)


def test_archived_response_format_is_explicit():
    text = json.dumps(record())
    with pytest.raises(ValidationError):
        parse_jsonl_response(text, 7)
    assert parse_jsonl_response(text, 7, response_format="task_record") == record()


def test_detector_validates_before_clipping():
    with pytest.raises(ValidationError):
        detector_record(7, [[0, 0, float("nan"), 10]], [0.8], (100, 100))
    with pytest.raises(ValidationError):
        detector_record(7, [[10, 10, 2, 2]], [0.8], (100, 100))
    with pytest.raises(ValidationError):
        detector_record(7, [[0, 0, 10, 10]], [1.1], (100, 100))
    with pytest.raises(ValidationError):
        detector_record(7, [[0, 0, 10, 10]], [], (100, 100))
    result, clipped = detector_record(7, [[-2, 0, 105, 100]], [0.8], (100, 100))
    assert clipped == 1
    assert result["boxes"] == [{"left": 0, "top": 0, "width": 1, "height": 1}]
    with pytest.raises(ValidationError):
        detector_record(7, [[-2, 0, 105, 100]], [0.8], (100, 100), clip=False)
