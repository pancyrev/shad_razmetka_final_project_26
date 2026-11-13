"""Micro object metrics with maximum-cardinality bipartite matching."""


def iou(a, b):
    intersection = max(0.0, min(a["left"] + a["width"], b["left"] + b["width"])
                       - max(a["left"], b["left"])) * max(
        0.0, min(a["top"] + a["height"], b["top"] + b["height"]) - max(a["top"], b["top"]))
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def maximum_matches(predicted, truth, threshold):
    adjacency = [[j for j, gt in enumerate(truth) if iou(pred, gt) >= threshold]
                 for pred in predicted]
    owners = {}

    def augment(pred_index, seen):
        for gt_index in adjacency[pred_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in owners or augment(owners[gt_index], seen):
                owners[gt_index] = pred_index
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def nms(boxes, scores, threshold):
    if threshold is None:
        return boxes, scores
    if not 0 <= threshold <= 1:
        raise ValueError("nms_iou must be in [0,1] or null")
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    keep = []
    for index in order:
        if all(iou(boxes[index], boxes[other]) <= threshold for other in keep):
            keep.append(index)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def evaluate(predictions, truth, thresholds=(0.5, 0.75), primary_iou=0.5):
    if set(predictions) != set(truth):
        raise ValueError("prediction IDs must exactly match ground-truth IDs")
    result = {}
    for threshold in thresholds:
        tp = fp = fn = negative_fp = negative_images_with_fp = 0
        error_images = 0
        for photo_id, gt in truth.items():
            record = predictions[photo_id]
            # Failures remain explicit records; all their GT objects count as FN.
            pred = record["boxes"] if record["error"] is None else []
            matched = maximum_matches(pred, gt, threshold)
            image_fp, image_fn = len(pred) - matched, len(gt) - matched
            tp += matched
            fp += image_fp
            fn += image_fn
            error_images += bool(image_fp or image_fn or record["error"] is not None)
            if not gt:
                negative_fp += image_fp
                negative_images_with_fp += image_fp > 0
        denominator = 2 * tp + fp + fn
        prefix = f"iou_{threshold:g}"
        values = {"tp": tp, "fp": fp, "fn": fn, "errors": fp + fn,
                  "images_with_errors": error_images,
                  "precision": tp / (tp + fp) if tp + fp else 0.0,
                  "recall": tp / (tp + fn) if tp + fn else 0.0,
                  "f1": 2 * tp / denominator if denominator else 0.0,
                  "negative_fp": negative_fp, "negative_images_with_fp": negative_images_with_fp}
        result.update({f"{prefix}/{key}": value for key, value in values.items()})
        if threshold == primary_iou:
            result.update(values)
    result["images"] = len(truth)
    result["negative_images"] = sum(not boxes for boxes in truth.values())
    result["processing_errors"] = sum(p["error"] is not None for p in predictions.values())
    return result

