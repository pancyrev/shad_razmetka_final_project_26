#!/usr/bin/env python3
"""Suggest a 400-photo sample and build a local gallery for human review."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter
from pathlib import Path


FIELDS = ["photo_id", "source_stratum", "observation_url", "photo_url", "purpose", "scenario", "notes"]


def suggest(rows: list[dict], plan: dict) -> dict[int, str]:
    chosen: dict[int, str] = {}
    for index, group in enumerate(plan["groups"]):
        eligible = [
            row for row in rows
            if row["source_stratum"] == group["name"]
            and min(row["image_width"] or 0, row["image_height"] or 0) >= 400
        ]
        count = group["target_count"] + group["training_exam_count"]
        if len(eligible) < count:
            raise ValueError(f"{group['name']}: only {len(eligible)} candidates at least 400 px; need {count}")
        selected = random.Random(plan["seed"] + 1000 + index).sample(eligible, count)
        for row in selected[:group["target_count"]]:
            chosen[row["photo_id"]] = "main"
        for row in selected[group["target_count"]:]:
            chosen[row["photo_id"]] = "worker_train_exam"
    return chosen


def write_csv(rows: list[dict], chosen: dict[int, str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "photo_id": row["photo_id"],
                "source_stratum": row["source_stratum"],
                "observation_url": row["observation_url"],
                "photo_url": row["photo_url"],
                "purpose": chosen.get(row["photo_id"], "reserve"),
                "scenario": "",
                "notes": "",
            })


def write_gallery(rows: list[dict], chosen: dict[int, str], plan: dict, path: Path) -> None:
    parts = ["""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Кандидаты для собственной выборки бабочек</title>
<style>
body{font:16px/1.4 system-ui,sans-serif;margin:24px;background:#f7f7f5;color:#202020}
h1{font-size:1.5rem}h2{margin:36px 0 12px}.hint{max-width:75ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
article{background:#fff;border:2px solid #ddd;border-radius:8px;padding:8px}
article.main{border-color:#177b58}article.worker_train_exam{border-color:#3265aa}
img{display:block;width:100%;height:180px;object-fit:contain;background:#eee}
.meta{font-size:13px;overflow-wrap:anywhere}.tag{font-weight:700}.main .tag{color:#177b58}.worker_train_exam .tag{color:#3265aa}
a{color:#0759b9}footer{margin-top:28px}
</style></head><body>
<h1>Кандидаты для собственной выборки</h1>
<p class="hint">Зелёная рамка — основная выборка для разметки и сравнения моделей; синяя — отдельные фото для обучения и экзамена исполнителей; серая — резерв. Категория iNaturalist не является истинной меткой. После просмотра исправьте столбец <code>purpose</code> в <code>selection_500_proposal.csv</code>, сохранив квоты групп.</p>
"""]
    for group in plan["groups"]:
        name = group["name"]
        group_rows = [row for row in rows if row["source_stratum"] == name]
        parts.append(f"<h2>{html.escape(name)} — {group['target_count']} основных + {group['training_exam_count']} учебных из {len(group_rows)}</h2><div class=grid>")
        for row in group_rows:
            purpose = chosen.get(row["photo_id"], "reserve")
            cls = f' class="{purpose}"'
            photo_id = row["photo_id"]
            source = html.escape(row["source_photo_url"], quote=True)
            original = html.escape(row["photo_url"], quote=True)
            observation = html.escape(row["observation_url"], quote=True)
            tag = f'<span class="tag">{html.escape(purpose)}</span>'
            parts.append(f'<article{cls}><a href="{original}" target="_blank" rel="noopener noreferrer"><img src="{source}" loading="lazy" alt="Фото {photo_id}"></a><div class="meta">{tag} · ID {photo_id}<br>{row["image_width"]} × {row["image_height"]}<br><a href="{observation}" target="_blank" rel="noopener noreferrer">Наблюдение</a> · <a href="{original}" target="_blank" rel="noopener noreferrer">Оригинал</a></div></article>')
        parts.append("</div>")
    parts.append("<footer>Источник: iNaturalist, лицензия каждого фото проверена как CC0. Список предложений — черновик до проверки изображений.</footer></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("candidates_700.jsonl"))
    parser.add_argument("--plan", type=Path, default=Path("sampling_plan.json"))
    parser.add_argument("--csv", type=Path, default=Path("selection_500_proposal.csv"))
    parser.add_argument("--gallery", type=Path, default=Path("candidate_gallery_500.html"))
    args = parser.parse_args()
    for path in (args.csv, args.gallery):
        if path.exists():
            raise FileExistsError(f"Won't overwrite existing review work: {path}")
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if len({row["photo_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate photo IDs are not unique")
    chosen = suggest(rows, plan)
    if len(chosen) != 500 or list(chosen.values()).count("main") != 400 or list(chosen.values()).count("worker_train_exam") != 100:
        raise ValueError(f"Expected 400 main and 100 training/exam photos, got {len(chosen)} total")
    write_csv(rows, chosen, args.csv)
    write_gallery(rows, chosen, plan, args.gallery)
    print(f"Proposed {len(chosen)} photos: {dict(Counter(chosen.values()))}")
    print(f"Review {args.gallery} and edit {args.csv} before freezing the final dataset")


if __name__ == "__main__":
    main()
