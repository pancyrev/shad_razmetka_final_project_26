import json

import pytest

from butterfly_bench.schema import ParseError, ValidationError, detector_record, parse_jsonl_response


def record(boxes=None):
    return {"photo_id": 7, "boxes": boxes or [], "error": None}


def test_successful_empty_and_nonempty_answers():
    assert parse_jsonl_response(json.dumps(record()), 7) == record()
    box = {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
    assert parse_jsonl_response(json.dumps(record([box])), 7)["boxes"] == [box]


@pytest.mark.parametrize("text", [
    '```json\n{"photo_id":7,"boxes":[],"error":null}\n```',
    '{"photo_id":7,"boxes":[],"error":null}\n{"photo_id":7,"boxes":[],"error":null}',
    '{"photo_id":7,"boxes":[],"boxes":[],"error":null}',
    '{"photo_id":7,"boxes":[NaN],"error":null}',
    'answer: {"photo_id":7,"boxes":[],"error":null}',
    '',
])
def test_bad_jsonl_is_not_silently_repaired(text):
    with pytest.raises(ParseError):
        parse_jsonl_response(text, 7)


@pytest.mark.parametrize("obj", [
    {"photo_id": 8, "boxes": [], "error": None},
    {"photo_id": True, "boxes": [], "error": None},
    {"photo_id": 7, "boxes": None, "error": None},
    {"photo_id": 7, "boxes": [], "error": "failed"},
    {"photo_id": 7, "boxes": []},
    record([{"left": 0.9, "top": 0, "width": 0.2, "height": 0.1}]),
    record([{"left": 0, "top": 0, "width": 0, "height": 0.1}]),
    record([{"left": 0, "top": 0, "width": True, "height": 0.1}]),
    record([{"left": 100, "top": 20, "width": 30, "height": 40}]),
    record([{"left": 0, "top": 0, "width": "0.1", "height": 0.1}]),
])
def test_schema_rejects_impossible_values(obj):
    with pytest.raises(ValidationError):
        parse_jsonl_response(json.dumps(obj), 7)


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

