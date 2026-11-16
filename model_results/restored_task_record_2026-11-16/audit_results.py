"""Audit the repeat against archived selected prompts and recompute metrics."""

import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from butterfly_bench.metrics import evaluate
from butterfly_bench.models import prompt_for
from butterfly_bench.schema import parse_jsonl_response, validate_record


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


rows = []
for model in ("qwen", "gemma"):
    current = HERE / model
    search = ROOT / "model_results/prompt_search_2026-11-15" / model
    best = read_json(search / "summary.json")[0]
    original = ROOT / best["output_dir"]
    cfg = OmegaConf.load(current / "config.yaml")
    old_cfg = OmegaConf.load(original / "config.yaml")
    assert OmegaConf.to_container(cfg.model) == OmegaConf.to_container(old_cfg.model)
    for field in ("data", "seed", "deterministic", "mode", "retries", "postprocess", "evaluation"):
        assert cfg[field] == old_cfg[field], field
    data = read_json(current / "data.json")
    assert data == read_json(original / "data.json")
    predictions = read_lines(current / "predictions.jsonl")
    assert [p["photo_id"] for p in predictions] == data["photo_ids"]
    assert len(predictions) == 100
    for record in predictions:
        if record["error"] is None:
            validate_record(record, record["photo_id"])
        else:
            assert set(record) == {"photo_id", "boxes", "error"}
            assert record["boxes"] is None and isinstance(record["error"], str)
    attempts = read_lines(current / "attempts.jsonl")
    feedback = {}
    for attempt in attempts:
        photo_id = attempt["photo_id"]
        assert attempt["seed"] == (cfg.seed + photo_id * 1009 + attempt["attempt"] - 1) % 2**32
        if "raw_text" in attempt:
            assert attempt["details"]["effective_prompt"] == prompt_for(
                cfg.model, photo_id, feedback.get(photo_id))
        if attempt["stage"] == "success":
            parsed = parse_jsonl_response(attempt["raw_text"], photo_id)
            assert parsed == next(p for p in predictions if p["photo_id"] == photo_id)
        else:
            feedback[photo_id] = attempt["error"]
    reference = {r["photo_id"]: r["boxes"] for r in read_lines(
        ROOT / "annotations/worker_reference_first_pass.jsonl")}
    truth = {photo_id: reference[photo_id] for photo_id in data["photo_ids"]}
    metrics = read_json(current / "metrics.json")
    recomputed = evaluate({p["photo_id"]: p for p in predictions}, truth)
    assert all(metrics[key] == value for key, value in recomputed.items())
    assert metrics["total_attempts"] == len(attempts)
    assert metrics["parsing_errors"] == sum(a["stage"] == "parsing" for a in attempts)
    assert metrics["validation_errors"] == sum(a["stage"] == "validation" for a in attempts)
    model_info = read_json(current / "model.json")
    if model == "qwen":
        assert model_info["device"] == "cuda:0" and not model_info["quantized"]
    else:
        assert model_info["gpu_layers_offloaded"] == 49
        assert model_info["quantization"] == "UD-Q4_K_XL"
    old_predictions = read_lines(original / "predictions.jsonl")
    changed_ids = [p["photo_id"] for p, old in zip(predictions, old_predictions, strict=True) if p != old]
    rows.append({"model": model, "original_run": best["output_dir"],
                 "images": 100, "attempts": len(attempts),
                 "same_prompt_weights_and_inference_settings": True,
                 "same_dataset_order_and_ground_truth": True,
                 "request_prompts_and_seeds_verified": True,
                 "full_record_parsing_and_metric_recomputation_verified": True,
                 "predictions_differing_from_original": changed_ids,
                 "metrics": metrics,
                 "sha256": {name: hashlib.sha256((current / name).read_bytes()).hexdigest()
                            for name in ("config.yaml", "attempts.jsonl", "predictions.jsonl", "metrics.json")}})
(HERE / "audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
table = []
for row in rows:
    original_metrics = read_json(ROOT / row["original_run"] / "metrics.json")
    for label, metrics in (("исходный", original_metrics), ("повтор", row["metrics"])):
        table.append(
            f"| {row['model']} / {label} | {metrics['iou_0.5/f1']:.6f} | "
            f"{metrics['iou_0.75/f1']:.6f} | {metrics['precision']:.6f} | "
            f"{metrics['recall']:.6f} | {metrics['errors']} | {metrics['processing_errors']} | "
            f"{metrics['parsing_errors']} / {metrics['validation_errors']} | "
            f"{metrics['total_attempts']} | {metrics['processing_seconds']:.1f} |"
        )
identical = all(not row["predictions_differing_from_original"] for row in rows)
report = """# Повторная проверка восстановленного формата

По просьбе пользователя восстановлены прежние промпты и парсер из `1a5ab6c`:
Qwen и Gemma получают photo_id и возвращают полный JSONL-объект
`{"photo_id":ID,"boxes":[...],"error":null}`. Проверка ID, координат и повторы
при ошибках сохранены. OWLv2 и DINO не изменялись.

Проверены `qwen_tuned` и `gemma_tuned` на прежних 100 validation-изображениях
(61 рамка, 42 отрицательных примера), seed=42. Ревизии моделей/датасета,
порядок изображений и настройки инференса совпадают с исходными лучшими
прогонами. Независимый split test не использовался. Модели запускались по
очереди под `/tmp/butterfly-prompt-search-gpu.lock` на RTX 3080.
Qwen использует Transformers/CUDA BF16 без квантования; Gemma — llama.cpp,
UD-Q4_K_XL, выгрузка 49/49 слоёв на GPU.

| Модель / прогон | F1@0.5 | F1@0.75 | Precision@0.5 | Recall@0.5 | FP+FN | Тех. отказы | Parsing / validation | Попытки | Обработка, с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(table) + "\n\n"
report += ("Все 200 итоговых предсказаний точно совпали с соответствующими исходными.\n"
           if identical else "Различающиеся ID перечислены в audit.json.\n")
report += """
Время включает обработку изображений и повторы, без загрузки модели, ожидания
GPU и завершения Comet. Ошибки парсинга относятся к попыткам, технические
отказы — к изображениям после исчерпания повторов.

32 теста и Ruff прошли. `audit_results.py` сверяет реальные запросы, сиды,
полные ответы, порядок ID и настройки с историческими артефактами, пересчитывает
метрики по сохранённым рамкам и создаёт `audit.json`.

## Воспроизведение

Из корня репозитория, последовательно:

```bash
flock /tmp/butterfly-prompt-search-gpu.lock .venv/bin/python model_results/restored_task_record_2026-11-16/validation_driver.py qwen
flock /tmp/butterfly-prompt-search-gpu.lock .venv/bin/python model_results/restored_task_record_2026-11-16/validation_driver.py gemma
.venv/bin/python model_results/restored_task_record_2026-11-16/audit_results.py
```

Сценарий читает закреплённый Parquet и веса из локального кеша. Перед новым
запуском нужен другой каталог результатов: существующие файлы не перезаписываются.
В каталогах моделей сохранены конфигурации, сырые ответы, предсказания, метрики,
тайминги, GPU metadata, submission при отсутствии окончательных отказов и
`comet.json` со ссылками на эксперименты. Режим Comet auto/online сохранён;
полнота облачной копии отдельно не проверялась.

[Подбор промптов](../prompt_search_2026-11-15/README.md) и
[эксперимент без служебных полей](../boxes_only_2026-11-16/README.md)
сохранены без изменения исходных ответов и метрик.
"""
(HERE / "README.md").write_text(report)
print(json.dumps([{r["model"]: {"changed_predictions": len(r["predictions_differing_from_original"]),
                              "metrics": r["metrics"]}} for r in rows], indent=2))
