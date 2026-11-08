#!/usr/bin/env python3
"""Collect CC0 iNaturalist photo candidates without downloading the images.

Source categories are sampling hints, not labels. Human bbox/review determines
whether any adult butterfly is actually visible in a selected photograph.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


API_URL = "https://api.inaturalist.org/v1/observations"
S3_ORIGIN = "https://inaturalist-open-data.s3.amazonaws.com"
PHOTO_PATH = re.compile(r"^/photos/(?P<id>\d+)/(?:square|small|medium|large|original)\.(?P<ext>jpg|jpeg|png)$", re.I)
REVIEW_FIELDS = ["photo_id", "source_stratum", "observation_url", "photo_url", "include", "scenario", "notes"]


def read_jsonl_ids(path: Path) -> set[int]:
    result: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            photo_id = row["photo_id"]
            if type(photo_id) is not int or photo_id in result:
                raise ValueError(f"{path}:{line_number}: invalid or repeated photo_id")
            result.add(photo_id)
    return result


def photo_to_record(observation: dict, photo: dict, stratum: str, retrieved_at: str) -> dict | None:
    if (photo.get("license_code") or "").lower() != "cc0":
        return None
    photo_id = photo.get("id")
    observation_id = observation.get("id")
    source_url = photo.get("url")
    if type(photo_id) is not int or type(observation_id) is not int or not isinstance(source_url, str):
        return None
    match = PHOTO_PATH.match(urlparse(source_url).path)
    if not match or int(match.group("id")) != photo_id:
        return None
    dimensions = photo.get("original_dimensions") or {}
    ext = match.group("ext").lower()
    taxon = observation.get("taxon") or {}
    return {
        "photo_id": photo_id,
        "photo_url": f"{S3_ORIGIN}/photos/{photo_id}/original.{ext}",
        "observation_id": observation_id,
        "observation_url": observation.get("uri") or f"https://www.inaturalist.org/observations/{observation_id}",
        "source_stratum": stratum,
        "taxon_id": taxon.get("id"),
        "taxon_name": taxon.get("name"),
        "license_code": "cc0",
        "attribution": photo.get("attribution"),
        "source_photo_url": source_url,
        "image_width": dimensions.get("width"),
        "image_height": dimensions.get("height"),
        "retrieved_at": retrieved_at,
    }


def page_url(group: dict, page: int) -> str:
    query = {
        "taxon_id": group["taxon_id"],
        "photos": "true",
        "photo_license": "cc0",
        "per_page": 200,
        "page": page,
    }
    if group.get("without_taxon_id"):
        query["without_taxon_id"] = group["without_taxon_id"]
    return f"{API_URL}?{urlencode(query)}"


def fetch_page(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "SHAD butterfly final project (public CC0 metadata)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            if not isinstance(result, dict) or not isinstance(result.get("results"), list):
                raise ValueError(f"Unexpected iNaturalist response: {url}")
            return result
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("API retry loop ended unexpectedly")


def collect_group(group: dict, exam_ids: set[int], seen_ids: set[int], seed: int, retrieved_at: str, dry_run: bool) -> list[dict]:
    desired = group["candidate_count"]
    pool: list[dict] = []
    pool_ids: set[int] = set()
    pages = group["pages"]
    if dry_run:
        print(f"{group['name']}: target {group['target_count']}, candidates {desired}")
        for page in pages:
            print(" ", page_url(group, page))
        return []

    for index, page in enumerate(pages):
        if index:
            time.sleep(1.1)  # iNaturalist recommends about one API request per second.
        payload = fetch_page(page_url(group, page))
        for observation in payload["results"]:
            taxon = observation.get("taxon") or {}
            ancestry = set(taxon.get("ancestor_ids") or [])
            ancestry.add(taxon.get("id"))
            if group.get("without_taxon_id") in ancestry:
                continue
            for photo in observation.get("photos") or []:
                record = photo_to_record(observation, photo, group["name"], retrieved_at)
                if record is None:
                    continue
                photo_id = record["photo_id"]
                if photo_id in exam_ids or photo_id in seen_ids or photo_id in pool_ids:
                    continue
                pool.append(record)
                pool_ids.add(photo_id)
                break  # At most one photo per observation.
        print(f"{group['name']}: {len(pool)} eligible candidates after page {page}")
        if len(pool) >= desired:
            break
    if len(pool) < desired:
        raise ValueError(f"{group['name']}: need {desired} candidates, found {len(pool)}; add pages to the plan")
    sample = random.Random(seed).sample(pool, desired)
    seen_ids.update(record["photo_id"] for record in sample)
    return sample


def write_outputs(records: list[dict], output: Path, review: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    review.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or review.exists():
        raise FileExistsError("Candidate outputs already exist; move them before starting a new collection")
    temp_output = output.with_name(output.name + ".tmp")
    temp_review = review.with_name(review.name + ".tmp")
    try:
        with temp_output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with temp_review.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow({**{key: record.get(key, "") for key in REVIEW_FIELDS}, "include": "", "scenario": "", "notes": ""})
        temp_output.replace(output)
        temp_review.replace(review)
    finally:
        temp_output.unlink(missing_ok=True)
        temp_review.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("sampling_plan.json"))
    parser.add_argument("--exam", type=Path, default=Path("dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("candidates_700.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("candidates_700_review.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    groups = plan["groups"]
    if not groups or sum(group["target_count"] for group in groups) != 400:
        raise ValueError("The sampling plan must have groups totaling 400 main photos")
    if sum(group.get("training_exam_count", 0) for group in groups) != 100:
        raise ValueError("The sampling plan must have groups totaling 100 worker training/exam photos")
    exam_ids = read_jsonl_ids(args.exam)
    if len(exam_ids) != 400:
        raise ValueError("Expected exactly 400 unique Contest photo IDs")
    if args.dry_run:
        for index, group in enumerate(groups):
            collect_group(group, exam_ids, set(), plan["seed"] + index, "", True)
        return
    if args.output.exists() or args.review.exists():
        raise FileExistsError("Candidate outputs already exist; move them before starting a new collection")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    seen_ids: set[int] = set()
    for index, group in enumerate(groups):
        if index:
            time.sleep(1.1)
        records.extend(collect_group(group, exam_ids, seen_ids, plan["seed"] + index, retrieved_at, False))
    write_outputs(records, args.output, args.review)
    print(f"Saved {len(records)} CC0 candidates to {args.output} and {args.review}")
    print("Review the candidates visually before freezing the 400 main and 100 worker training/exam photographs.")


if __name__ == "__main__":
    main()
