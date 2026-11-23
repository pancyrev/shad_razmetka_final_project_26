import json
import random
from pathlib import Path

from PIL import Image

from scripts import pipeline_sampling
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


def test_download_similarity_and_manifest_steps_write_their_own_artifacts(tmp_path, monkeypatch):
    data = Path(__file__).resolve().parents[1] / "data"
    run_dir = tmp_path / "run"
    plan = json.loads((data / "sampling/sampling_plan.json").read_text())
    candidates = collect_candidates(
        run_dir, plan, jsonl_rows(data / "contest_400/photos.jsonl"),
        snapshot=data / "sampling/runs/20261108T001420_435829Z/candidates_from_api.jsonl",
    )
    selection = select_photos(run_dir, candidates, plan)
    selected = [dict(row, image_width=12, image_height=12) for row in selection]

    def write_test_image(row, target, max_file_bytes, remaining_bytes):
        pixels = random.Random(row["photo_id"]).randbytes(12 * 12 * 3)
        Image.frombytes("RGB", (12, 12), pixels).save(target)

    monkeypatch.setattr(pipeline_sampling, "download", write_test_image)
    audited = pipeline_sampling.download_originals(run_dir, selected, tmp_path / "images")
    assert jsonl_rows(run_dir / "download_audit.jsonl") == audited
    assert not (run_dir / "similarity_review.csv").exists()
    assert not (run_dir / "main/photos.jsonl").exists()

    pairs = pipeline_sampling.review_similar_photos(run_dir, audited)
    assert (run_dir / "similarity_review.csv").is_file()
    assert not (run_dir / "reference/photos.jsonl").exists()

    report = pipeline_sampling.save_photo_manifests(run_dir, audited)
    assert report["similarity_pairs_for_review"] == pairs
    for dirname, size in (("main", 400), ("reference", 100)):
        path = run_dir / dirname / "photos.jsonl"
        manifest = jsonl_rows(path)
        assert len(manifest) == size
        assert (path.parent / manifest[0]["image_path"]).is_file()
