"""Strict task schema. Syntactic validity does not imply a correct detection."""

import json
import math
from numbers import Real


class PredictionError(ValueError):
    stage = "validation"


class ParseError(PredictionError):
    stage = "parsing"


class ValidationError(PredictionError):
    stage = "validation"


def validate_photo_id(value):
    if type(value) is not int or value < 0:
        raise ValidationError("photo_id must be a nonnegative integer")
    return value


def validate_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValidationError("boxes must be an array")
    result = []
    fields = {"left", "top", "width", "height"}
    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or set(box) != fields:
            raise ValidationError(f"box {i}: expected exactly {sorted(fields)}")
        for key, value in box.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValidationError(f"box {i}: {key} must be a finite number")
        b = {key: float(value) for key, value in box.items()}
        if (not 0 <= b["left"] <= 1 or not 0 <= b["top"] <= 1
                or not 0 < b["width"] <= 1 or not 0 < b["height"] <= 1):
            raise ValidationError(f"box {i}: invalid origin or nonpositive size")
        # Permit only floating-point addition roundoff, not out-of-image coordinates.
        if b["left"] + b["width"] > 1 + 1e-12 or b["top"] + b["height"] > 1 + 1e-12:
            raise ValidationError(f"box {i}: extends beyond normalized image boundaries")
        result.append(b)
    return result


def validate_record(record, photo_id):
    if not isinstance(record, dict) or set(record) != {"photo_id", "boxes", "error"}:
        raise ValidationError("record must have exactly photo_id, boxes, error")
    validate_photo_id(record["photo_id"])
    if record["photo_id"] != photo_id:
        raise ValidationError(f"photo_id must equal {photo_id}")
    if record["error"] is not None:
        raise ValidationError("a successful model response must have error=null")
    return {"photo_id": photo_id, "boxes": validate_boxes(record["boxes"]), "error": None}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ParseError(f"non-JSON numeric constant: {value}")


def parse_jsonl_response(text, photo_id, *, response_format="boxes_only"):
    """Parse model-owned fields and attach trusted metadata from the input.

    task_record is retained only to reproduce archived prompt experiments.
    """
    if not isinstance(text, str) or len(text.strip().splitlines()) != 1:
        raise ParseError("expected exactly one nonempty JSONL line")
    try:
        obj = json.loads(text.strip(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ParseError(str(exc)) from exc
    if response_format == "task_record":
        return validate_record(obj, photo_id)
    if response_format != "boxes_only":
        raise ValueError(f"unknown VLM response format: {response_format}")
    if not isinstance(obj, dict) or set(obj) != {"boxes"}:
        raise ValidationError("model response must have exactly one field: boxes")
    return validate_record({"photo_id": photo_id, "boxes": obj["boxes"], "error": None}, photo_id)


def detector_record(photo_id, boxes_xyxy, scores, image_size, *, clip=True):
    """Validate raw detector tensors before their explicit pixel clipping policy."""
    width, height = image_size
    if len(boxes_xyxy) != len(scores):
        raise ValidationError("detector box/score counts differ")
    boxes, clipped = [], 0
    for i, (coords, score) in enumerate(zip(boxes_xyxy, scores, strict=True)):
        if len(coords) != 4 or not all(math.isfinite(x) for x in coords):
            raise ValidationError(f"detector box {i}: nonfinite or malformed coordinates")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError(f"detector score {i}: expected a finite probability")
        x1, y1, x2, y2 = coords
        if x1 >= x2 or y1 >= y2:
            raise ValidationError(f"detector box {i}: reversed or empty bounds")
        if clip:
            bounded = [min(width, max(0, x1)), min(height, max(0, y1)),
                       min(width, max(0, x2)), min(height, max(0, y2))]
            clipped += bounded != list(coords)
            x1, y1, x2, y2 = bounded
        boxes.append({"left": x1 / width, "top": y1 / height,
                      "width": (x2 - x1) / width, "height": (y2 - y1) / height})
    return validate_record({"photo_id": photo_id, "boxes": boxes, "error": None}, photo_id), clipped
