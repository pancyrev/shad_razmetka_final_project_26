#!/usr/bin/env python3
"""Build a simple, self-contained HTML gallery with red butterfly boxes.

Requires Pillow. The HTML contains all image previews and needs no server.
"""

from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path


TEMPLATE = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка бабочек</title>
<style>
body{font:16px/1.4 system-ui,sans-serif;margin:24px;background:#f7f7f5;color:#202020}
h1{font-size:1.5rem}.hint{max-width:75ch}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
article{background:#fff;border:2px solid #ddd;border-radius:8px;padding:8px}
.thumbnail{height:270px;display:flex;align-items:center;justify-content:center;background:#eee}
.photo{position:relative;display:block;width:100%;line-height:0}
img{display:block;width:100%;height:auto}
.boxes{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.boxes rect{fill:none;stroke:#e11;stroke-width:2px;vector-effect:non-scaling-stroke}
.meta{font-size:13px;overflow-wrap:anywhere;margin-top:6px}
a{color:#0759b9}footer{margin-top:28px;font-size:13px}
</style></head><body>
<h1>Разметка бабочек</h1>
<p class="hint">__TOTAL__ фотографий · __BOXES__ рамка. Красными рамками выделены бабочки.</p>
<div class="grid">
__CARDS__
</div>
<footer>Разметка: <a href="annotations/worker_reference_first_pass.json">JSON</a> · <a href="annotations/worker_reference_first_pass.jsonl">JSONL</a></footer>
</body></html>
'''


def main() -> None:
    from PIL import Image, ImageOps

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=root/'annotations/worker_reference_first_pass.json')
    parser.add_argument('--images', type=Path, default=root/'images')
    parser.add_argument('--output', type=Path, default=root/'worker_reference_annotated_gallery.html')
    parser.add_argument('--max-size', type=int, default=1024)
    args = parser.parse_args()
    rows = json.loads(args.annotations.read_text(encoding='utf-8'))
    assert len({row['photo_id'] for row in rows}) == len(rows), 'Duplicate photo IDs'
    cards = []
    for row in rows:
        pid, boxes = row['photo_id'], row['boxes']
        assert type(pid) is int
        assert row['photo_url'].startswith('https://inaturalist-open-data.s3.amazonaws.com/photos/')
        for box in boxes:
            assert set(box) == {'left', 'top', 'width', 'height'}
            assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in box.values())
            assert 0 <= box['left'] < 1 and 0 <= box['top'] < 1
            assert 0 < box['width'] <= 1 and 0 < box['height'] <= 1
            assert box['left'] + box['width'] <= 1 and box['top'] + box['height'] <= 1
        with Image.open(args.images/f'{pid}.jpg') as original:
            preview = ImageOps.exif_transpose(original).convert('RGB')
            preview.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)
            width, height = preview.size
            buffer = BytesIO()
            preview.save(buffer, format='JPEG', quality=84, optimize=True, progressive=True)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        rects = ''.join(
            f'<rect x="{b["left"]}" y="{b["top"]}" width="{b["width"]}" height="{b["height"]}"></rect>'
            for b in boxes
        )
        url = escape(row['photo_url'], quote=True)
        # Keep the overlay on the displayed photo, excluding any letterboxing.
        max_width = 270 * width / height
        cards.append(
            f'<article data-id="{pid}"><div class="thumbnail">'
            f'<a class="photo" style="max-width:{max_width:.6f}px" href="{url}" target="_blank" rel="noopener noreferrer">'
            f'<img src="data:image/jpeg;base64,{encoded}" width="{width}" height="{height}" loading="lazy" alt="Фото {pid}">'
            f'<svg class="boxes" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">{rects}</svg></a></div>'
            f'<div class="meta"><b>{pid}</b> · рамок: {len(boxes)}<br>'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">Оригинал</a></div></article>'
        )
    total_boxes = sum(len(row['boxes']) for row in rows)
    word = 'рамка' if total_boxes % 10 == 1 and total_boxes % 100 != 11 else 'рамки' if total_boxes % 10 in (2, 3, 4) and total_boxes % 100 not in (12, 13, 14) else 'рамок'
    document = TEMPLATE.replace('__TOTAL__', str(len(rows))).replace('__BOXES__', str(total_boxes))
    document = document.replace(' рамка.', f' {word}.').replace('__CARDS__', '\n'.join(cards))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding='utf-8')
    print(json.dumps({'output':str(args.output), 'photos':len(rows), 'boxes':total_boxes,
                      'bytes':args.output.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
