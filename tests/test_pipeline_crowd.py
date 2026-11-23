from unittest.mock import Mock

import pytest

from scripts.pipeline_crowd import CrowdRun, _answer
from scripts.pipeline_tasks_api import Receipts


def test_rectangle_from_actual_editor_shape():
    values = {"photo_id": 15, "image": "https://example.com/15.jpg"}
    assignment = {"id": "a", "user_id": "u"}
    points = {"p1": {"x": 0.625, "y": 0.75}, "p2": {"x": 0.125, "y": 0.25}}
    answer = {"status": "FOUND", "annotations": [{"type": "rectangle", "data": points}]}
    parsed = _answer(values, assignment, answer)
    assert parsed["status"] == "FOUND"
    assert parsed["boxes"] == [{"left": 0.125, "top": 0.25, "width": 0.5, "height": 0.5}]
    points["p1"]["x"] = 1.25
    assert _answer(values, assignment, answer)["status"] == "UNCERTAIN"


def test_retry_recovers_lost_post_without_duplicate_pool(tmp_path):
    api = Mock()
    body = {"project_id": "project", "private_name": "unique", "reward_per_assignment": 2}
    api.request.side_effect = ConnectionError("response lost")
    path = tmp_path / "receipts.json"
    with pytest.raises(ConnectionError):
        Receipts(path, api, 1).resource("work", "pools", body)

    api.items.return_value = []
    with pytest.raises(RuntimeError, match="Исход создания"):
        Receipts(path, api, 1).resource("work", "pools", body)
    api.items.return_value = [dict(body, id="existing")]
    assert Receipts(path, api, 1).resource("work", "pools", body) == "existing"
    assert Receipts(path, api, 1).resource("work", "pools", body) == "existing"
    api.request.assert_called_once()
    with pytest.raises(ValueError, match="изменились"):
        Receipts(path, api, 1).resource("work", "pools", dict(body, reward_per_assignment=3))


def test_revised_truth_cannot_reuse_paid_campaign(tmp_path):
    main = [{"photo_id": 100, "photo_url": "https://example.com/100"}]
    reference = [{"photo_id": i, "photo_url": f"https://example.com/{i}", "boxes": []} for i in range(35)]
    with CrowdRun(tmp_path, main, reference, api=Mock())._stage():
        pass
    reference[0]["boxes"] = [{"left": 0.125, "top": 0.25, "width": 0.5, "height": 0.5}]
    with pytest.raises(ValueError, match="изменились"), CrowdRun(tmp_path, main, reference, api=Mock())._stage():
        pass


def test_separate_notebook_stages_reject_foreign_results_before_api_calls(tmp_path):
    api = Mock()
    main = [{"photo_id": 100, "photo_url": "https://example.com/100"}]
    reference = [{"photo_id": i, "photo_url": f"https://example.com/{i}", "boxes": []} for i in range(35)]
    crowd = CrowdRun(tmp_path, main, reference, api=api)
    with pytest.raises(ValueError, match="Результат экзамена"):
        crowd.annotate(qualification={"project_id": "another-project"})
    with pytest.raises(ValueError, match="Учебный пул"):
        crowd.open_reviewer_exam(training_id="another-training")
    (crowd.folder / "annotation_work.jsonl").write_text("")
    with pytest.raises(ValueError, match="Экзамен ревьюеров"):
        crowd.review({}, exam_id="another-exam")
    assert not api.mock_calls
