"""Ten prompt-only Qwen experiments, with an audited adaptation after variant 3."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from prompt_search_common import run_search

BASELINE = OmegaConf.load(
    Path(__file__).resolve().parents[1] / "butterfly_bench/conf/model/qwen.yaml"
).output_prompt

SCHEMA = (
    'Return exactly one JSON line with exactly the keys "photo_id", "boxes", "error". '
    'Use the supplied integer photo_id and error:null. boxes is an array; each box is '
    'an object with ALL FOUR named numeric fields "left", "top", "width", "height". '
    'All coordinates are fractions of the full image width/height, with origin at '
    'the top left. left>=0, top>=0, width>0, height>0, left+width<=1, top+height<=1. '
    'No butterflies means boxes:[]. No Markdown or other text.'
)
TARGET = (
    "Locate every adult butterfly, including skippers, with one tight rectangular "
    "box enclosing the complete visible wings, body, antennae and legs. Exclude "
    "moths, caterpillars, pupae, other insects and background objects. For cropped "
    "or occluded butterflies include only the visible parts. "
)

PROMPTS = [
    {"name": "baseline", "description": "Original detailed production prompt.",
     "prompt": str(BASELINE)},
    {"name": "concise", "description": "Short localization instruction without an example.",
     "prompt": (
         "Detect all adult butterflies (including skippers, excluding moths). "
         "Give one tight box around the entire visible insect, including wings. "
         'Output one JSON line: {"photo_id":INTEGER,"boxes":[{"left":NUMBER,'
         '"top":NUMBER,"width":NUMBER,"height":NUMBER}],"error":null}. '
         "Replace INTEGER with the supplied photo_id and NUMBER with decimal fractions "
         "of image width/height in [0,1]. left/top locate the top-left corner; "
         "width/height are sizes, not bottom-right coordinates. Boxes must stay inside "
         "the image. If absent use an empty boxes array. Output JSON only."
     )},
    {"name": "json_exemplars", "description": "Explicit positive and negative valid JSON examples.",
     "prompt": TARGET + (
         "Use this output format exactly, replacing the example photo_id and coordinates "
         "with the actual image values. Example with one butterfly: "
         '{"photo_id":123,"boxes":[{"left":0.12,"top":0.23,"width":0.56,'
         '"height":0.41}],"error":null}. Example with no butterflies: '
         '{"photo_id":123,"boxes":[],"error":null}. '
         "Every number in a box needs its own field name; do not use unnamed lists "
         "of coordinates. Use fractions, not pixels. left/top are the top-left "
         "position and width/height are the full box size. "
     ) + SCHEMA},
    {"name": "corners_then_sizes", "description": "Internally locate opposite corners before conversion.",
     "prompt": TARGET + (
         "First locate the top-left and bottom-right corners of each butterfly's "
         "tight bounding rectangle. Work in relative coordinates: x increases from "
         "0 at the image's left edge to 1 at its right edge, y from 0 at the top to "
         "1 at the bottom. Internally estimate x_min,y_min,x_max,y_max. Then output "
         "left=x_min, top=y_min, width=x_max-x_min, height=y_max-y_min. "
         "Do NOT output x_max as width or y_max as height. Do not show intermediate work. "
     ) + SCHEMA},
    {"name": "four_edges", "description": "Identify separate outermost edges and avoid center/size confusion.",
     "prompt": TARGET + (
         "For each butterfly, inspect its leftmost visible point, highest point, "
         "rightmost point and lowest point. Use those four outer edges, not the "
         "center or a single wing. The full box must cover the entire butterfly and "
         "as little background as possible. left is the fraction of image width "
         "BEFORE the insect, top is the fraction of image height ABOVE it. width "
         "is the fraction of image width OCCUPIED by the whole insect, height the "
         "fraction of image height OCCUPIED by it. These are four separate named values. "
     ) + SCHEMA},
    {"name": "internal_1000", "description": "Use Qwen's common 0–1000 grounding scale internally, then normalize.",
     "prompt": TARGET + (
         "For accurate visual grounding, internally locate every butterfly using "
         "a 0–1000 image coordinate system: top-left=(0,0), bottom-right=(1000,1000). "
         "Estimate its bounding corners (x1,y1,x2,y2), then convert them before "
         "answering: left=x1/1000, top=y1/1000, width=(x2-x1)/1000, "
         "height=(y2-y1)/1000. Only output the resulting decimal fractions. "
         "Never output the 0–1000 coordinates or intermediate calculations. "
     ) + SCHEMA},
    {"name": "percentage_geometry", "description": "Percent-of-image mental geometry converted to fractions.",
     "prompt": TARGET + (
         "Estimate what percentage of the FULL image lies to the left of the "
         "butterfly, above it, and what percentage of the FULL image its total "
         "width and height occupy. Divide those percentages by 100 to obtain "
         "left, top, width, height. For example, a box occupying 60% of image "
         "width must have width 0.60, not 0.30 or 60. Look carefully at wings "
         "extending sideways; do not restrict the box to the body or one wing. "
         "Do all calculations silently. "
     ) + SCHEMA},
    {"name": "russian", "description": "Russian task and geometry instructions with unchanged field names.",
     "prompt": (
         "Найди всех взрослых дневных бабочек, включая толстоголовок. Исключи ночных "
         "бабочек, гусениц, куколок и других насекомых. Для каждой бабочки найди тесную "
         "прямоугольную рамку вокруг ВСЕХ видимых крыльев, тела, усиков и лапок. "
         "Сначала мысленно определи крайние левую, верхнюю, правую и нижнюю границы. "
         "Верни одну строку корректного JSON с полями photo_id (заданный целый ID), "
         "boxes (массив рамок), error (null). У каждой рамки ровно четыре именованных "
         "числовых поля: left, top, width, height. left — левая граница, делённая на "
         "ширину изображения; top — верхняя граница, делённая на высоту; width — "
         "полная ширина рамки, делённая на ширину изображения; height — полная высота "
         "рамки, делённая на высоту изображения. Не путай ширину с правой границей. "
         "Используй доли от 0 до 1: left>=0, top>=0, width>0, height>0, "
         "left+width<=1, top+height<=1. Если бабочек нет, boxes должен быть []. "
         "Никаких пояснений, Markdown или дополнительных полей."
     )},
    {"name": "chinese", "description": "Chinese localization and strict JSON instructions.",
     "prompt": (
         "请找出图像中所有成年蝴蝶（包括弄蝶），排除飞蛾、毛毛虫、蛹和其他昆虫。"
         "每只蝴蝶只画一个紧密的轴对齐矩形框，包含所有可见的翅膀、身体、触角和足。"
         "先在心里确定整只蝴蝶的最左、最上、最右、最下边界，再计算框的位置和大小。"
         "只输出一行合法JSON，且只有photo_id、boxes、error三个键。photo_id使用给定的整数ID，"
         "error必须为null。boxes是数组；每个框必须包含left、top、width、height四个命名数值字段。"
         "left是左边界除以整张图宽度，top是上边界除以整张图高度，width是框的总宽度除以图宽，"
         "height是框的总高度除以图高。width不是右边界，height不是下边界。"
         "所有坐标必须是0到1的小数，left>=0，top>=0，width>0，height>0，"
         "left+width<=1，top+height<=1。没有蝴蝶时boxes为[]。不要输出解释、Markdown或中间步骤。"
     )},
    {"name": "silent_checklist", "description": "Internal identification, complete-object geometry, then syntax verification.",
     "prompt": (
         "You are a precise butterfly-detection annotator. Before answering, silently "
         "perform these checks: (1) Count all visible adult butterflies, including "
         "skippers; moths and immature life stages do not count. (2) For every "
         "butterfly, locate its complete visible silhouette including both wings, "
         "body and appendages. Place one tight box around that entire silhouette. "
         "(3) Estimate left and right edges as fractions of the FULL image width, "
         "top and bottom edges as fractions of its FULL height; subtract opposite "
         "edges to obtain width and height. (4) Check that no target is missed or "
         "duplicated, no box covers just a fragment, and no box extends outside "
         "the image. (5) Check that every JSON coordinate has its own quoted field "
         "name, that all braces balance, and that error is null. "
         "Do not output the checklist or reasoning. "
     ) + SCHEMA},
]

# Variants 2 and 3 repeatedly added Markdown fences. Keep their full results,
# and strengthen only the seven variants that have not yet been evaluated.
# The original and revised lists, hashes and reason are preserved in the results.
INITIAL_PROMPTS = [dict(prompt) for prompt in PROMPTS]
BASELINE_OUTPUT_CONTRACT = str(BASELINE)[str(BASELINE).index("Return exactly one line"):]
for variant in PROMPTS[3:]:
    original = variant["prompt"]
    prefix = original[:-len(SCHEMA)] if original.endswith(SCHEMA) else original + "\n"
    variant["prompt"] = (
        prefix + BASELINE_OUTPUT_CONTRACT
        + " The first character of your response must be { and the last character must be }. "
        "Do not wrap the JSON in triple backticks or code fences."
    )


def write_report(rows):
    root = Path(__file__).resolve().parents[1]
    target = root / "model_results/prompt_search_2026-11-15/qwen"
    baseline = next(row for row in rows if row["name"] == "baseline")
    best = rows[0]
    previous = json.loads((root / "model_results/validation_2026-11-13/qwen/metrics.json").read_text())
    lines = [
        "# Qwen: перебор десяти промптов",
        "",
        ("Исходные десять вариантов сохранены в `prompts_initial.json`. После полных "
        "прогонов 1–3 остальные семь адаптированы: их геометрические стратегии "
        "сохранены, а инструкция формата заменена на точный контракт исходного baseline "
        "с дополнительным явным запретом code fences и требованием начать с `{` и "
        "закончить `}`. Причина — массовые ответы с Markdown-обёрткой в вариантах 2–3. "
        "Уже выполненные 1–3 не менялись. Финальный список — `prompts.json`, "
        "время, причины и SHA256 ревизии — `prompt_revision.json`. "
        "Ровно десять разных вариантов проверены по одному разу на полном наборе. Каждый проверен "
        "на всех тех же 100 изображениях validation `xartik/inaturalist_butterflies` "
        "(61 эталонная рамка, 58 положительных и 42 отрицательных изображения). "
        "Порядок, данные и ревизии модели совпадают с исходным прогоном. "
        "`Qwen/Qwen3-VL-2B-Instruct` работает через Transformers в BF16 без квантования. "
        "Seed=42; генерация, разрешение изображения, строгий парсер, проверка схемы и "
        "до трёх попыток с обратной связью остаются прежними. Меняется только текст промпта."),
        "",
        ("Главный критерий — F1@IoU 0.5, затем F1@IoU 0.75, затем меньше окончательных "
        "ошибок обработки. Время ниже — сумма обработки 100 изображений без ожидания "
        "GPU-блокировки и без загрузки модели. `parsing` и `validation` учитывают "
        "все отклонённые попытки; `failed` — изображения без допустимого ответа после повторов."),
        "",
        "| Промпт | F1@0.5 | F1@0.75 | Precision@0.5 | Recall@0.5 | failed | parsing | validation | Попытки | Время, с |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        relative = Path(row["output_dir"]).name
        lines.append(
            f"| [{row['name']}]({relative}/metrics.json) | {row['iou_0.5/f1']:.6f} | "
            f"{row['iou_0.75/f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['processing_errors']} | {row['parsing_errors']} | {row['validation_errors']} | "
            f"{row['total_attempts']} | {row['processing_seconds']:.1f} |"
        )
    lines.append("")
    for name in ["concise", "json_exemplars"]:
        row = next(row for row in rows if row["name"] == name)
        attempts = [json.loads(line) for line in
                    (root / row["output_dir"] / "attempts.jsonl").read_text().splitlines()]
        fenced = sum((attempt.get("raw_text") or "").lstrip().startswith("```")
                     for attempt in attempts)
        lines.append(
            f"`{name}`: {fenced} из {len(attempts)} ответов начинались с Markdown code fence; "
            f"окончательно отклонено {row['processing_errors']} изображений. "
            "Парсер оставлен строгим: такие ответы не исправлялись автоматически."
        )
        lines.append("")
    russian = next(row for row in rows if row["name"] == "russian")
    russian_predictions = [json.loads(line) for line in
                           (root / russian["output_dir"] / "predictions.jsonl").read_text().splitlines()]
    empty_count = sum(record["boxes"] == [] for record in russian_predictions)
    lines.append(
        f"`russian`: {empty_count}/100 ответов содержали пустой boxes; "
        f"failed={russian['processing_errors']}, F1@0.5={russian['iou_0.5/f1']:.6f}. "
        "Допустимый JSON и геометрически возможные координаты сами по себе не означают "
        "правильное обнаружение объектов."
    )
    lines.extend([
        "",
        (f"Лучший вариант: **{best['name']}**. F1@0.5 изменился с "
        f"{baseline['iou_0.5/f1']:.6f} у повторённого baseline до "
        f"{best['iou_0.5/f1']:.6f} (разница "
        f"{best['iou_0.5/f1'] - baseline['iou_0.5/f1']:+.6f}). "
        "Точный текст сохранён в `best_prompt.txt` и `best_prompt.json`."),
        "",
        (f"Исторический baseline: F1@0.5={previous['iou_0.5/f1']:.6f}, "
        f"F1@0.75={previous['iou_0.75/f1']:.6f}, failed={previous['processing_errors']}. "
        f"Повторённый baseline: F1@0.5={baseline['iou_0.5/f1']:.6f}, "
        f"F1@0.75={baseline['iou_0.75/f1']:.6f}, failed={baseline['processing_errors']}."),
        "",
        ("В каждом каталоге варианта сохранены config.yaml, predictions.jsonl, "
        "attempts.jsonl (все сырые ответы и причины отклонения), timings.jsonl, "
        "metrics.json, gpu.jsonl, environment.json и comet.json. "
        "GPU-сессии обеих моделей сериализует общий flock; расписание находится "
        "в ../gpu_schedule.jsonl. Новые веса и большие данные не скачивались. "
        "Comet ограничивал частоту загрузки артефактов: облачная копия может быть "
        "неполной; локальные файлы сохранены полностью, а SDK при необходимости "
        "сохраняет offline-архив для последующей синхронизации."),
        "",
        ("Это результаты подбора промпта на validation, а не независимая оценка "
        "качества. Выбор лучшего из десяти повышает риск подгонки к этим 100 "
        "изображениям; для оценки переноса нужна отдельная выборка. "
        "Основные конфиги модели не изменялись."),
        "",
        "## Варианты",
        "",
    ])
    lines.extend(f"- `{p['name']}`: {p['description']}" for p in PROMPTS)
    (target / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_report(run_search("qwen", PROMPTS))
