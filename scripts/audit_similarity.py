#!/usr/bin/env python3
"""Flag visually similar downloaded photos for human duplicate review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps


def dhash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.get_flattened_data())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("download_audit.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("similarity_review.csv"))
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Won't overwrite human review: {args.output}")
    rows = [json.loads(line) for line in args.audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    hashes = [(row["photo_id"], row["purpose"], dhash(Path(row["local_path"]))) for row in rows]
    pairs = []
    for i, (left_id, left_purpose, left_hash) in enumerate(hashes):
        for right_id, right_purpose, right_hash in hashes[i + 1:]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= args.threshold:
                pairs.append((left_id, right_id, distance, left_purpose, right_purpose))
    pairs.sort(key=lambda pair: (pair[2], pair[0], pair[1]))
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["photo_id_a", "photo_id_b", "dhash_distance", "purpose_a", "purpose_b", "same_photo_yes_no", "notes"])
        for pair in pairs:
            writer.writerow([*pair, "", ""])
    print(f"Compared {len(rows)} images; {len(pairs)} pairs need human review in {args.output}")


if __name__ == "__main__":
    main()
