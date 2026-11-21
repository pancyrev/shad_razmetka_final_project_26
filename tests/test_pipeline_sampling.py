import json
from pathlib import Path

from scripts.download_selected import jsonl_rows
from scripts.pipeline_sampling import collect_candidates, select_photos


def test_original_seed_recreates_the_archived_selection(tmp_path):
    root = Path(__file__).resolve().parents[1] / "data"
    archive = root / "sampling/runs/20261108T001420_435829Z"
    plan = json.loads((root / "sampling/sampling_plan.json").read_text())
    assert plan["seed"] == 20261107
    candidates = collect_candidates(
        tmp_path, plan, jsonl_rows(root / "contest_400/photos.jsonl"),
        snapshot=archive / "candidates_from_api.jsonl",
    )
    selected = select_photos(tmp_path, candidates, plan)
    assert len(candidates) == 700
    assert (tmp_path / "selection_500_proposal.csv").read_bytes() == (archive / "selection_500_proposal.csv").read_bytes()
    for purpose, directory in (("main", "golden_400"), ("worker_train_exam", "reference_100")):
        actual = {r["photo_id"]: r["photo_url"] for r in selected if r["purpose"] == purpose}
        expected = {r["photo_id"]: r["photo_url"] for r in jsonl_rows(root / directory / "photos.jsonl")}
        assert actual == expected
