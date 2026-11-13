import pytest

from butterfly_bench.metrics import evaluate, maximum_matches, nms


def box(left=0, width=0.6):
    return {"left": left, "top": 0, "width": width, "height": 1}


def pred(boxes, error=None):
    return {"boxes": boxes, "error": error}


def test_matching_reassigns_pair_where_greedy_fails():
    # First prediction matches both GT boxes. Second only matches the first.
    truth = [box(0), box(0.4)]
    assert maximum_matches([box(0.2), box(0)], truth, 0.5) == 2


def test_micro_metrics_failures_negatives_and_duplicate_predictions():
    truth = {1: [box()], 2: [], 3: [box()], 4: []}
    predictions = {1: pred([box(), box()]), 2: pred([box()]),
                   3: pred(None, "decode error"), 4: pred([])}
    metrics = evaluate(predictions, truth)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 2, 1)
    assert metrics["f1"] == pytest.approx(0.4)
    assert metrics["negative_fp"] == 1
    assert metrics["processing_errors"] == 1
    assert metrics["errors"] == 3
    assert metrics["images_with_errors"] == 3


def test_iou_boundary_and_empty_dataset_convention():
    assert maximum_matches([box(0, 1)], [box(0, 0.5)], 0.5) == 1
    assert maximum_matches([box(0, 1)], [box(0, 0.5)], 0.75) == 0
    assert evaluate({1: pred([])}, {1: []})["f1"] == 0
    with pytest.raises(ValueError):
        evaluate({1: pred([])}, {2: []})


def test_nms_is_optional_and_stable():
    boxes, scores = [box(), box(), box(0.4)], [0.2, 0.8, 0.6]
    assert nms(boxes, scores, None) == (boxes, scores)
    assert nms(boxes, scores, 0.5) == ([boxes[1], boxes[2]], [0.8, 0.6])

