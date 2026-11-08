#!/usr/bin/env python3
"""Download and verify proposed CC0 originals, with a 1 GiB total safety cap.

The produced JSONL files remain drafts until people review the photographs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


PURPOSES = {"main", "worker_train_exam"}
S3_HOST = "inaturalist-open-data.s3.amazonaws.com"


class DownloadLimitError(Exception):
    pass


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_records(candidates: Path, selection: Path, exam: Path, plan: Path) -> list[dict]:
    by_id = {row["photo_id"]: row for row in jsonl_rows(candidates)}
    exam_ids = {row["photo_id"] for row in jsonl_rows(exam)}
    sampling_plan = json.loads(plan.read_text(encoding="utf-8"))
    with selection.open(encoding="utf-8", newline="") as stream:
        proposal = list(csv.DictReader(stream))
    if {int(row["photo_id"]) for row in proposal} != set(by_id) or len(proposal) != len(by_id):
        raise ValueError("Selection CSV must contain every candidate exactly once")
    chosen: list[dict] = []
    for entry in proposal:
        purpose = entry["purpose"]
        if purpose not in PURPOSES | {"reserve"}:
            raise ValueError(f"Unknown purpose for photo {entry['photo_id']}")
        if purpose == "reserve":
            continue
        photo_id = int(entry["photo_id"])
        record = by_id.get(photo_id)
        if record is None or photo_id in exam_ids or record["license_code"] != "cc0":
            raise ValueError(f"Invalid selected photo: {photo_id}")
        if record["photo_url"] != entry["photo_url"] or record["source_stratum"] != entry["source_stratum"]:
            raise ValueError(f"Candidate metadata mismatch for photo {photo_id}")
        chosen.append({**record, "purpose": purpose, "scenario": entry.get("scenario", ""), "notes": entry.get("notes", "")})
    counts = Counter(row["purpose"] for row in chosen)
    if counts != {"main": 400, "worker_train_exam": 100}:
        raise ValueError(f"Need 400 main and 100 worker training/exam photos, got {dict(counts)}")
    for group in sampling_plan["groups"]:
        for purpose, expected in [("main", group["target_count"]), ("worker_train_exam", group["training_exam_count"])]:
            actual = sum(row["source_stratum"] == group["name"] and row["purpose"] == purpose for row in chosen)
            if actual != expected:
                raise ValueError(f"{group['name']} / {purpose}: need {expected}, got {actual}")
    return chosen


def image_path(row: dict, image_dir: Path) -> Path:
    parsed = urlparse(row["photo_url"])
    if parsed.scheme != "https" or parsed.netloc != S3_HOST or parsed.query or parsed.fragment:
        raise ValueError(f"Unexpected photo URL for ID {row['photo_id']}")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Unexpected image format for ID {row['photo_id']}")
    return image_dir / f"{row['photo_id']}{suffix}"


def verify_image(path: Path, expected_width: int, expected_height: int) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError(f"Unexpected decoded format: {image.format}")
        dimensions = image.size
        image.verify()
    if dimensions != (expected_width, expected_height):
        raise ValueError(f"Dimensions mismatch: expected {(expected_width, expected_height)}, got {dimensions}")
    return dimensions


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(row: dict, target: Path, max_file_bytes: int, remaining_bytes: int) -> None:
    partial = target.with_name(target.name + ".part")
    request = Request(row["photo_url"], headers={"User-Agent": "SHAD butterfly final project (CC0 photo verification)"})
    for attempt in range(3):
        partial.unlink(missing_ok=True)
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 200 or not response.headers.get("Content-Type", "").startswith("image/"):
                    raise ValueError("URL did not return an image")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > min(max_file_bytes, remaining_bytes):
                    raise DownloadLimitError("Image would exceed the file or total size cap")
                size = 0
                with partial.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > min(max_file_bytes, remaining_bytes):
                            raise DownloadLimitError("Image exceeded the file or total size cap")
                        stream.write(chunk)
            verify_image(partial, row["image_width"], row["image_height"])
            partial.replace(target)
            return
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnidentifiedImageError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(2 ** attempt)
        except DownloadLimitError:
            partial.unlink(missing_ok=True)
            raise


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("candidates_700.jsonl"))
    parser.add_argument("--selection", type=Path, default=Path("selection_500_proposal.csv"))
    parser.add_argument("--exam", type=Path, default=Path("dataset.jsonl"))
    parser.add_argument("--plan", type=Path, default=Path("sampling_plan.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke test with first N selected photos; 0 means all 500")
    parser.add_argument("--max-total-mib", type=int, default=1024)
    parser.add_argument("--max-file-mib", type=int, default=20)
    args = parser.parse_args()
    chosen = selected_records(args.candidates, args.selection, args.exam, args.plan)
    selection_hash = checksum(args.selection)
    selected = chosen[:args.limit] if args.limit else chosen
    args.image_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(path.stat().st_size for path in args.image_dir.iterdir() if path.is_file())
    limit_bytes = args.max_total_mib * 1024 * 1024
    max_file_bytes = args.max_file_mib * 1024 * 1024
    audited: list[dict] = []
    errors: list[dict] = []
    hashes: dict[str, int] = {}
    for index, row in enumerate(selected, 1):
        target = image_path(row, args.image_dir)
        try:
            if not target.exists():
                download(row, target, max_file_bytes, limit_bytes - total_bytes)
                total_bytes += target.stat().st_size
            verify_image(target, row["image_width"], row["image_height"])
            digest = checksum(target)
            if digest in hashes:
                errors.append({"photo_id": row["photo_id"], "error": "exact_duplicate", "other_photo_id": hashes[digest]})
            hashes[digest] = row["photo_id"]
            audited.append({**row, "local_path": str(target), "bytes": target.stat().st_size, "sha256": digest})
        except Exception as error:
            if isinstance(error, HTTPError):
                label = f"HTTP_{error.code}"
            else:
                label = type(error).__name__
            errors.append({"photo_id": row["photo_id"], "error": label})
            if isinstance(error, DownloadLimitError):
                print(f"Stopped at photo {row['photo_id']}: configured size cap reached", flush=True)
                break
        if index % 25 == 0 or index == len(selected):
            print(f"Checked {index}/{len(selected)}; verified {len(audited)}; errors {len(errors)}; stored {total_bytes / 1024**2:.1f} MiB", flush=True)

    if checksum(args.selection) != selection_hash:
        raise RuntimeError("Selection CSV changed during download; rerun against the current selection")
    write_jsonl(Path("download_audit.jsonl"), audited)
    write_jsonl(Path("download_errors.jsonl"), errors)
    if errors:
        raise SystemExit(f"{len(errors)} issues found. See download_errors.jsonl; replace failed/duplicate photos from reserve")
    if args.limit:
        print(f"Pilot download passed for {len(audited)} images")
        return
    if len(audited) != 500:
        raise RuntimeError(f"Only {len(audited)} of 500 images were verified")
    write_jsonl(Path("main_dataset_draft.jsonl"), [{"photo_id": row["photo_id"], "photo_url": row["photo_url"]} for row in audited if row["purpose"] == "main"])
    write_jsonl(Path("worker_training_exam_draft.jsonl"), [{"photo_id": row["photo_id"], "photo_url": row["photo_url"]} for row in audited if row["purpose"] == "worker_train_exam"])
    print("All 500 originals verified; draft URL manifests are ready for human review")


if __name__ == "__main__":
    main()
