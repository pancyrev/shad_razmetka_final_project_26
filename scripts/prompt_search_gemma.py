"""Ten Gemma prompt candidates; adaptive revision is preserved beside run artifacts."""

import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASELINE = OmegaConf.load(ROOT / "model_results/validation_2026-11-13/gemma/config.yaml").model.prompt

SCHEMA = (
    'Output exactly one line of valid JSON, without Markdown or explanation. '
    'Use exactly {"photo_id":THE_INTEGER_ID,"boxes":[{"left":X,"top":Y,"width":W,"height":H}],'
    '"error":null}. Each box has exactly the four named numeric fields. '
    'Use normalized fractions of image dimensions, never pixels or a 0-1000 scale. '
    'The top-left of the image is (0,0), bottom-right is (1,1). '
    'left=x_min/image_width; top=y_min/image_height; '
    'width=(x_max-x_min)/image_width; height=(y_max-y_min)/image_height. '
    'All numbers must be finite. left>=0, top>=0, width>0, height>0, '
    'left+width<=1 and top+height<=1. If no adult butterflies are present use "boxes": []. '
    'Copy the supplied photo_id as an integer and set error to null. '
)

TARGET = (
    'Locate every adult butterfly. Exclude moths, caterpillars, pupae, other insects, '
    'flowers and shadows. Return one tight axis-aligned rectangle for each butterfly, '
    'including all its visible wings, body, antennae and legs. For occluded or cropped '
    'animals, box the visible part only. '
)

PROMPTS = [
    {"name": "01_baseline", "description": "Original prompt rerun without changes.",
     "prompt": BASELINE},
    {"name": "02_compact", "description": "Compact direct localization instruction and exact schema.",
     "prompt": TARGET + SCHEMA},
    {"name": "03_edges_then_extents", "description": "Derive width and height from four visual extrema.",
     "prompt": TARGET +
     'Work silently in this order: identify the butterfly, locate the leftmost visible tip, '
     'rightmost visible tip, highest visible tip and lowest visible tip, then convert those '
     'four edges to left, top, width and height. Width is a horizontal span, not the right '
     'edge; height is a vertical span, not the bottom edge. Check every rectangle against '
     'the actual wing outline before answering. ' + SCHEMA},
    {"name": "04_visual_scan", "description": "Systematic image scan to find small or multiple adults.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'Act as an expert annotator of butterfly photographs. Scan the entire image '
     'from top to bottom and left to right, including corners and background. Count the adult '
     'butterflies, then localize each one exactly once. A butterfly may have open or closed '
     'wings, be small, side-on, occluded, or cut off by the image border. Do not duplicate '
     'the same animal or combine two animals in one rectangle.'},
    {"name": "05_classification_first", "description": "Explicit adult butterfly versus moth decision.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'First silently decide which visible animals are adult butterflies. '
     'Moths, including large butterfly-like moths, are not target objects. Use the whole '
     'animal and its anatomy rather than color alone: clubbed antennae support butterfly '
     'identification, while feathery antennae and a moth-like body support exclusion. '
     'Closed wings are allowed and a dull brown butterfly is still a butterfly. '
     'Only after deciding the target animals, draw their boxes.'},
    {"name": "06_box_verification", "description": "Silent final checks for tightness and normalized bounds.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'Before submitting, silently verify each box: its left edge touches the leftmost '
     'visible part, top edge the highest part, right edge the rightmost part, and bottom '
     'edge the lowest part. Remove unnecessary background padding but keep all visible '
     'appendages. Check that the box covers both wings when both are visible. Check the '
     'sums left+width and top+height: neither can exceed 1. If an edge is at the image '
     'border, the corresponding normalized edge is exactly 0 or 1.'},
    {"name": "07_fractional_ruler", "description": "Interpret normalized coordinates using fractional ruler landmarks.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'Use the image itself as a ruler: the left/top border is 0.0, one quarter across '
     'or down is 0.25, the center is 0.5, three quarters is 0.75, the right/bottom '
     'border is 1.0. Estimate the four outer edges using these landmarks, then subtract '
     'left from right for width and top from bottom for height. Use three decimal '
     'places when useful. Do not confuse the horizontal and vertical directions.'},
    {"name": "08_json_contract_first", "description": "Schema precedes task; explicit example with correct dimensions.",
     "prompt": SCHEMA +
     'For example, an object occupying x=20% through 70% and y=10% through 60% '
     'would have {"left":0.2,"top":0.1,"width":0.5,"height":0.5}. '
     'This is an example of arithmetic only, not a proposed box for this image. '
     'The output boxes must come from this image. ' + TARGET},
    {"name": "09_precision_focus", "description": "Conservative target decision, especially on negative images.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'This detection task includes photographs with no adult butterflies. '
     'An empty result is correct when the image shows a moth, larva, pupa, another insect '
     'or only plants. Do not invent a butterfly to satisfy the request. Do not box '
     'butterfly-shaped flowers, leaves, shadows, detached decorations or background '
     'patterns. Require a visible animal identifiable as an adult butterfly.'},
    {"name": "10_recall_and_tightness", "description": "Avoid missing folded, partial, or small butterflies and refine boxes.",
     "prompt": BASELINE + '\nAdditional annotation guidance: '
     'Annotate all adult butterflies in this image, including butterflies '
     'whose wings are folded, butterflies seen from the side, small butterflies, and '
     'partially hidden or cropped butterflies. Do not require bright colors or open '
     'wings. After finding each animal, refine the rectangle to its exact visible '
     'extent: use the outer wing tips and any visible antennae or legs. Avoid a generic '
     'square centered on the body; adapt width and height to the actual silhouette. '
     'Moths and non-adult life stages must still be excluded.'},
]


def build_report():
    """Populate the final report only after all ten full runs have completed."""
    base = ROOT / "model_results/prompt_search_2026-11-15/gemma"
    rows = []
    for index, prompt in enumerate(PROMPTS, 1):
        run_dir = base / f"{index:02d}_{prompt['name']}"
        metrics = json.loads((run_dir / "metrics.json").read_text())
        status = json.loads((run_dir / "status.json").read_text())["status"]
        if metrics["images"] != 100 or status not in {"complete", "completed_with_errors"}:
            raise ValueError(f"Incomplete run: {run_dir}")
        rows.append({"prompt": prompt, "metrics": metrics, "run_dir": run_dir})
    best = max(rows, key=lambda row: (row["metrics"]["iou_0.5/f1"],
                                     row["metrics"]["iou_0.75/f1"],
                                     -row["metrics"]["processing_errors"]))
    initial = rows[0]["metrics"]
    winner = best["metrics"]
    report_path = base / "report.md"
    methodology = report_path.read_text().split("## Текущий статус")[0].split("## Результаты")[0]
    table = ["## Результаты", "", "Все десять вариантов обработали по 100 изображений.", "",
             ("| Вариант | F1@0.5 | F1@0.75 | Precision@0.5 | Recall@0.5 | FP+FN | "
             "Технические ошибки | Отклонения JSON / схемы | Время обработки, с |"),
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        table.append(f"| `{row['prompt']['name']}` | {m['iou_0.5/f1']:.6f} | "
                     f"{m['iou_0.75/f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | "
                     f"{m['errors']} | {m['processing_errors']} | "
                     f"{m['parsing_errors']} / {m['validation_errors']} | {m['processing_seconds']:.1f} |")
    table.extend(["", (f"Выбран **{best['prompt']['name']}**: "
                  f"F1@0.5 = {winner['iou_0.5/f1']:.6f}, "
                  f"F1@0.75 = {winner['iou_0.75/f1']:.6f}. "
                  f"Разница с заново измеренным baseline: "
                  f"{winner['iou_0.5/f1'] - initial['iou_0.5/f1']:+.6f} и "
                  f"{winner['iou_0.75/f1'] - initial['iou_0.75/f1']:+.6f} соответственно."), "",
                  ("Технические ошибки — окончательные отказы после всех попыток. "
                  "Отклонения JSON и схемы считаются по всем попыткам, включая исправленные. "
                  "Неверно распознанные объекты и неточные допустимые рамки учитываются через FP/FN."), "",
                  ("Время относится к обработке изображений с повторами, без загрузки модели, "
                  "ожидания GPU lock и доставки в Comet."), "", "## Лучший промпт", "", "```text",
                  best["prompt"]["prompt"], "```", "",
                  ("К каждому запросу пайплайн добавляет `The photo_id for this image is <ID>.`; "
                  "при отказе — сообщение валидатора. Полный текст каждой попытки сохранён "
                  "в соответствующем `attempts.jsonl`."), "", "## Артефакты и ограничения", "",
                  ("Каталоги вариантов содержат `config.yaml`, `attempts.jsonl`, "
                  "`predictions.jsonl`, `metrics.json`, `data.json`, `model.json`, "
                  "`gpu.jsonl`, `llama-server.log`, `run.log` и данные Comet. "
                  "Общая сводка — `summary.json`, лучший текст — `best_prompt.txt/json`."), "",
                  ("Comet использовал режим auto/online, но сервер ограничивал загрузку. "
                  "SDK сохранял offline ZIP для недоставленных данных. Полнота облачной "
                  "копии не подтверждена; локальные метрики и сырые ответы сохранены полностью. "
                  "Fallback первого запуска может находиться в `.cometml-runs/` в корне "
                  "репозитория, последующих — в `<variant>/comet-offline/`."), "",
                  ("Первая попытка загрузки baseline завершилась CUDA OOM до обработки "
                  "изображений из-за дополнительной ROS задачи на GPU. Её диагностика сохранена "
                  "в `startup_failures/2026-11-16T102234Z_baseline_oom/` и не включается в "
                  "таблицу. После разрешённой пользователем остановки ROS все варианты "
                  "запускались с исходными параметрами."), ""])
    table.extend(["## Причины отклонений по всем попыткам", "",
                  ("Столбцы ниже взаимоисключающие: если в рамке есть значение >1, "
                  "она попадает в первый столбец, даже если одновременно выходит за границы."), "",
                  ("| Вариант | Координата >1 | Отрицательное начало / неположительный размер | "
                  "Сумма краёв >1 | Прочая схема | Парсинг | Runtime |"),
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        counts = Counter()
        attempts = [json.loads(line) for line in (row["run_dir"] / "attempts.jsonl").read_text().splitlines()]
        for attempt in attempts:
            stage = attempt["stage"]
            if stage == "success":
                continue
            if stage != "validation":
                counts[stage] += 1
                continue
            record = json.loads(attempt["raw_text"])
            values = [value for box in record.get("boxes", []) for value in box.values()
                      if isinstance(value, (int, float))]
            if any(value > 1 for value in values):
                counts["above_one"] += 1
            elif "invalid origin or nonpositive size" in attempt.get("error", ""):
                counts["sign"] += 1
            elif "extends beyond normalized image boundaries" in attempt.get("error", ""):
                counts["sum"] += 1
            else:
                counts["other"] += 1
        table.append(f"| `{row['prompt']['name']}` | {counts['above_one']} | {counts['sign']} | "
                     f"{counts['sum']} | {counts['other']} | {counts['parsing']} | {counts['runtime']} |")
    contract = rows[7]["metrics"]
    compact = rows[1]["metrics"]
    table.extend(["", (f"Вариант08 с числовым примером и schema-first порядком дал "
                  f"F1@0.5={contract['iou_0.5/f1']:.6f}, F1@0.75={contract['iou_0.75/f1']:.6f}, "
                  f"{contract['validation_errors']} отклонений схемы и "
                  f"{contract['processing_errors']} окончательных ошибок. "
                  f"У компактного02 — {compact['iou_0.5/f1']:.6f}/{compact['iou_0.75/f1']:.6f}, "
                  f"{compact['validation_errors']} и {compact['processing_errors']} соответственно. "
                  "Это сравнение пакетов инструкций: порядок текста и пример менялись одновременно, "
                  "поэтому отдельный причинный эффект числового примера этим перебором не установлен."), ""])
    report_path.write_text(methodology + "\n".join(table))


if __name__ == "__main__":
    from prompt_search_common import run_search

    run_search("gemma", PROMPTS)
    build_report()
