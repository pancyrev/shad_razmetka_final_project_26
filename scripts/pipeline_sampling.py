"""Fresh iNaturalist collection and manual reference annotation for pipeline.ipynb."""

import csv
import fcntl
import json
import os
import re
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from butterfly_bench.schema import validate_boxes, validate_photo_id
from scripts.audit_similarity import dhash
from scripts.collect_candidates import collect_group
from scripts.download_selected import (
    checksum,
    download,
    image_path,
    jsonl_rows,
    selected_records,
    verify_image,
)
from scripts.prepare_review import suggest, write_csv, write_gallery


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".sampling.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


@contextmanager
def _atomic(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _store(path, content):
    with _atomic(path) as tmp:
        tmp.write_text(content, encoding="utf-8")


def _jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)


def _freeze(path, content):
    """Do not mix different inputs, or invalidate annotations already in progress."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Входы этого запуска изменились ({path}). Начните новый запуск с первой ячейки.")
    else:
        _store(path, content)


def _indexed(rows):
    by_id = {validate_photo_id(row["photo_id"]): row for row in rows}
    if len(by_id) != len(rows) or len({row["photo_url"] for row in rows}) != len(rows):
        raise ValueError("В списке фотографий повторяются ID или URL")
    return by_id


def collect_candidates(run_dir, plan, contest_rows, *, snapshot=None):
    """Fetch the public API or replay its frozen candidate records, with the same checks."""
    run_dir = Path(run_dir)
    ids = set(_indexed(contest_rows))
    excluded_urls = {row["photo_url"] for row in contest_rows}
    groups = plan["groups"]
    names = [group["name"] for group in groups]
    if (len(ids) != 400 or len(set(names)) != len(names) or type(plan["seed"]) is not int
            or sum(g["target_count"] for g in groups) != 400
            or sum(g["training_exam_count"] for g in groups) != 100):
        raise ValueError("Проверьте план отбора и манифест 400 конкурсных фотографий.")
    snapshot = Path(snapshot) if snapshot is not None else None
    replay = jsonl_rows(snapshot) if snapshot is not None else None
    if replay is not None and Counter(r["source_stratum"] for r in replay) != {
        group["name"]: group["candidate_count"] for group in groups
    }:
        raise ValueError("Страты снимка API расходятся с планом отбора.")
    source = {"api": "live"} if snapshot is None else {"api_snapshot_sha256": checksum(snapshot)}
    with _lock(run_dir):
        _freeze(run_dir / "candidate_source.json", json.dumps(source) + "\n")
        _freeze(run_dir / "sampling_plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        _freeze(run_dir / "contest_photos.jsonl", _jsonl(contest_rows))
        rows, seen = [], set()
        for index, group in enumerate(groups):
            group_file = run_dir / "sampling" / f"group_{index:02}.jsonl"
            if replay is not None:
                batch = [row for row in replay if row["source_stratum"] == group["name"]]
            elif group_file.is_file():
                batch = jsonl_rows(group_file)
            else:
                if index:
                    time.sleep(1.1)
                batch = collect_group(group, ids, set(seen), plan["seed"] + index,
                                      datetime.now(UTC).isoformat(), False)
            batch_ids = set(_indexed(batch))
            if (len(batch) != group["candidate_count"] or batch_ids & (ids | seen)
                    or any(r["photo_url"] in excluded_urls or r["source_stratum"] != group["name"]
                           or r["license_code"] != "cc0" for r in batch)):
                raise ValueError(f"Некорректные кандидаты в группе {group['name']}")
            _freeze(group_file, _jsonl(batch))
            rows.extend(batch)
            seen.update(batch_ids)
        _indexed(rows)
        _freeze(run_dir / "candidates_from_api.jsonl", _jsonl(rows))
    return rows


def select_photos(run_dir, candidates, plan, *, suggested=None):
    """Suggest 400 + 100 using notebook 01's seed; honor saved manual CSV edits."""
    run_dir = Path(run_dir)
    proposal = run_dir / "selection_500_proposal.csv"
    archive = run_dir / "candidates_from_api.jsonl"
    with _lock(run_dir):
        if candidates != jsonl_rows(archive) or plan != json.loads((run_dir / "sampling_plan.json").read_text()):
            raise ValueError("План или список кандидатов отличается от этого запуска.")
        if not proposal.exists():
            if suggested is None:
                suggested = suggest(candidates, plan)
            with _atomic(proposal) as tmp:
                write_csv(candidates, suggested, tmp)
        selected = selected_records(archive, proposal, run_dir / "contest_photos.jsonl", run_dir / "sampling_plan.json")
        for row in selected:
            photo_id = row["photo_id"]
            if not re.fullmatch(rf"https://inaturalist-open-data\.s3\.amazonaws\.com/photos/{photo_id}/original\.(jpg|jpeg|png)", row["photo_url"]):
                raise ValueError(f"Неверный адрес оригинала для фото {photo_id}")
        with _atomic(run_dir / "candidate_gallery_500.html") as tmp:
            write_gallery(candidates, {r["photo_id"]: r["purpose"] for r in selected}, plan, tmp)
    return selected


def _similarity_report(run_dir, hashes):
    """Keep human decisions when the same image pairs are encountered on resume."""
    path = run_dir / "similarity_review.csv"
    decisions = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            decisions = {(r["photo_id_a"], r["photo_id_b"]): r for r in csv.DictReader(stream)}
    fields = ["photo_id_a", "photo_id_b", "dhash_distance", "purpose_a", "purpose_b", "same_photo_yes_no", "notes"]
    pairs = []
    for index, (a, purpose_a, hash_a) in enumerate(hashes):
        for b, purpose_b, hash_b in hashes[index + 1:]:
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= 4:
                record = decisions.get((str(a), str(b)), {})
                pairs.append([a, b, distance, purpose_a, purpose_b,
                              record.get("same_photo_yes_no", ""), record.get("notes", "")])
    with _atomic(path) as tmp, tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(sorted(pairs, key=lambda pair: (pair[2], pair[0], pair[1])))
    return len(pairs)


def download_originals(run_dir, selected, image_dir, *, expected_hashes=None, max_total_mib=1024, max_file_mib=20):
    """Download the selected images, verify dimensions and bytes, and write the audit JSONL."""
    run_dir, image_dir = Path(run_dir).resolve(), Path(image_dir).resolve()
    if image_dir == run_dir:
        raise ValueError("Укажите отдельный каталог для оригиналов.")
    expected = {"main": 400, "worker_train_exam": 100}
    if Counter(row["purpose"] for row in selected) != expected:
        raise ValueError("Ожидаются 400 основных и 100 учебных фотографий.")
    selected_ids = set(_indexed(selected))
    if expected_hashes is not None and set(expected_hashes) != selected_ids:
        raise ValueError("ID в исходном журнале SHA-256 не совпадают с выбранными фотографиями.")
    audited, content_ids = [], {}
    with _lock(run_dir), _lock(image_dir):
        prior = run_dir / "download_audit.jsonl"
        prior_rows = jsonl_rows(prior) if prior.is_file() else []
        if prior_rows:
            expected_files = {r["photo_id"]: (r["purpose"], image_path(r, image_dir)) for r in selected}
            previous_files = {r["photo_id"]: (r["purpose"], (run_dir / r["local_path"]).resolve())
                              for r in prior_rows}
            if expected_files != previous_files:
                raise ValueError("Для другого набора или каталога оригиналов начните новый запуск с первой ячейки.")
        prior_hashes = {r["photo_id"]: r["sha256"] for r in prior_rows}
        new_bytes = 0
        for index, row in enumerate(selected, 1):
            target = image_path(row, image_dir)
            if not target.exists():
                download(row, target, max_file_mib * 2**20, max_total_mib * 2**20 - new_bytes)
                new_bytes += target.stat().st_size
            verify_image(target, row["image_width"], row["image_height"])
            digest = checksum(target)
            pid = row["photo_id"]
            if expected_hashes is not None and digest != expected_hashes[pid]:
                raise ValueError(f"Фото {pid} не совпадает по SHA-256 с оригиналом исторического запуска.")
            if pid in prior_hashes and prior_hashes[pid] != digest:
                raise ValueError(f"Оригинал {pid} изменился относительно журнала этого запуска")
            if digest in content_ids:
                raise ValueError(f"Одинаковые оригиналы {pid} и {content_ids[digest]}: замените фото из резерва.")
            content_ids[digest] = pid
            audited.append({**row, "local_path": os.path.relpath(target, run_dir),
                            "sha256": digest, "bytes": target.stat().st_size})
            if index % 50 == 0:
                print(f"Проверено оригиналов {index}/{len(selected)}")
        _store(prior, _jsonl(audited))
    return audited


def review_similar_photos(run_dir, audit):
    """Compare hashes of downloaded images and save candidate pairs for visual review."""
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        if audit != jsonl_rows(run_dir / "download_audit.jsonl"):
            raise ValueError("Проверьте оригиналы перед поиском похожих фотографий.")
        hashes = [(r["photo_id"], r["purpose"], dhash(run_dir / r["local_path"])) for r in audit]
        return _similarity_report(run_dir, hashes)


def save_photo_manifests(run_dir, audit):
    """Publish both photo manifests and the sampling report after the similarity check."""
    run_dir = Path(run_dir).resolve()
    with _lock(run_dir):
        if audit != jsonl_rows(run_dir / "download_audit.jsonl"):
            raise ValueError("Манифесты должны строиться по журналу текущей загрузки.")
        with (run_dir / "similarity_review.csv").open(encoding="utf-8", newline="") as stream:
            pairs = sum(1 for _ in csv.DictReader(stream))
        counts = Counter(r["purpose"] for r in audit)
        for purpose, dirname in (("main", "main"), ("worker_train_exam", "reference")):
            path = run_dir / dirname / "photos.jsonl"
            rows = [{"photo_id": r["photo_id"], "photo_url": r["photo_url"],
                     "image_path": os.path.relpath(run_dir / r["local_path"], path.parent)}
                    for r in audit if r["purpose"] == purpose]
            _freeze(path, _jsonl(rows))
        report = {"candidate_count": len(jsonl_rows(run_dir / "candidates_from_api.jsonl")),
                  "main_count": counts["main"], "reference_count": counts["worker_train_exam"],
                  "similarity_pairs_for_review": pairs, "plan_sha256": checksum(run_dir / "sampling_plan.json"),
                  "selection_sha256": checksum(run_dir / "selection_500_proposal.csv"),
                  "main_sha256": checksum(run_dir / "main/photos.jsonl"),
                  "reference_sha256": checksum(run_dir / "reference/photos.jsonl")}
        _store(run_dir / "sampling_audit.json", json.dumps(report, indent=2) + "\n")
    return report


def download_photos(run_dir, selected, image_dir, *, expected_hashes=None, max_total_mib=1024, max_file_mib=20):
    """Download, check similarity and write manifests together."""
    audit = download_originals(run_dir, selected, image_dir, expected_hashes=expected_hashes,
                               max_total_mib=max_total_mib, max_file_mib=max_file_mib)
    pairs = review_similar_photos(run_dir, audit)
    save_photo_manifests(run_dir, audit)
    print(f"Пары с расстоянием dHash ≤ 4: {pairs}. Таблица: {Path(run_dir) / 'similarity_review.csv'}")
    return audit


def prepare_reference(run_dir):
    """Create a curator worksheet exactly once. Null = unreviewed; [] = verified absence."""
    refs = Path(run_dir) / "reference"
    path = refs / "labels.jsonl"
    with _lock(refs):
        photos = jsonl_rows(refs / "photos.jsonl")
        if not path.exists():
            _store(path, _jsonl([{"photo_id": r["photo_id"], "photo_url": r["photo_url"], "boxes": None}
                                 for r in photos]))
    return path


def _reference_rows(run_dir):
    refs = Path(run_dir) / "reference"
    photos = jsonl_rows(refs / "photos.jsonl")
    labels = _indexed(jsonl_rows(refs / "labels.jsonl"))
    if set(labels) != set(_indexed(photos)):
        raise ValueError("Лист разметки должен содержать те же ID, что и reference/photos.jsonl.")
    result = []
    for photo in photos:
        row = labels[photo["photo_id"]]
        if row["photo_url"] != photo["photo_url"] or "boxes" not in row:
            raise ValueError(f"Проверьте URL и поле boxes у {photo['photo_id']}")
        boxes = validate_boxes(row["boxes"]) if row["boxes"] is not None else None
        result.append({"photo_id": photo["photo_id"], "photo_url": photo["photo_url"], "boxes": boxes})
    return result


def review_reference(run_dir):
    """Visually review pending photos in Jupyter; save each decision immediately."""
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    from matplotlib.ticker import MultipleLocator

    path = prepare_reference(run_dir)
    rows = _reference_rows(run_dir)
    refs = path.parent
    photos = _indexed(jsonl_rows(refs / "photos.jsonl"))
    for index, row in enumerate(rows, 1):
        if row["boxes"] is not None:
            continue
        pid = row["photo_id"]
        with Image.open(refs / photos[pid]["image_path"]) as source:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(source, extent=(0, 1, 1, 0), aspect="auto")
            ax.set_box_aspect(source.height / source.width)
            ax.xaxis.tick_top()
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.grid(alpha=0.35)
            ax.set_title(f"{index}/{len(rows)} · фото {pid}", pad=25)
            display(fig)
            plt.close(fig)
        print("Введите для каждой взрослой бабочки left top width height от 0 до 1. "
              "Несколько рамок разделяйте точкой с запятой. none = бабочек нет, pause = прервать.")
        while True:
            answer = input("Рамки: ").strip()
            if answer.lower() == "pause":
                raise InterruptedError(f"Ответы сохранены в {path}. Продолжите с этой ячейки в том же ядре.")
            try:
                boxes = [] if answer.lower() == "none" else [
                    dict(zip(("left", "top", "width", "height"), map(float, item.split()), strict=True))
                    for item in answer.split(";")]
                boxes = validate_boxes(boxes)
            except ValueError as exc:
                print(f"Проверьте координаты: {exc}")
                continue
            with _lock(refs):
                current = _reference_rows(run_dir)
                existing = next(r for r in current if r["photo_id"] == pid)
                if existing["boxes"] is not None and existing["boxes"] != boxes:
                    raise ValueError(f"Для {pid} уже сохранён другой ответ. Перечитайте лист разметки.")
                existing["boxes"] = boxes
                _store(path, _jsonl(current))
            break
        clear_output(wait=True)


def load_reference(run_dir):
    """Validate all curator answers against this run and freeze the downstream task input."""
    refs = Path(run_dir) / "reference"
    with _lock(refs):
        result = _reference_rows(run_dir)
        pending = sum(row["boxes"] is None for row in result)
        if pending:
            raise ValueError(f"Осталось разметить {pending} фото в {refs / 'labels.jsonl'}.")
        _freeze(refs / "annotations.jsonl", _jsonl(result))
    return result
