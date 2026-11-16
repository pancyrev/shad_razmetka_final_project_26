# Qwen: перебор десяти промптов

Исходные десять вариантов сохранены в `prompts_initial.json`. После полных прогонов 1–3 остальные семь адаптированы: их геометрические стратегии сохранены, а инструкция формата заменена на точный контракт исходного baseline с дополнительным явным запретом code fences и требованием начать с `{` и закончить `}`. Причина — массовые ответы с Markdown-обёрткой в вариантах 2–3. Уже выполненные 1–3 не менялись. Финальный список — `prompts.json`, время, причины и SHA256 ревизии — `prompt_revision.json`. Ровно десять разных вариантов проверены по одному разу на полном наборе. Каждый проверен на всех тех же 100 изображениях validation `xartik/inaturalist_butterflies` (61 эталонная рамка, 58 положительных и 42 отрицательных изображения). Порядок, данные и ревизии модели совпадают с исходным прогоном. `Qwen/Qwen3-VL-2B-Instruct` работает через Transformers в BF16 без квантования. Seed=42; генерация, разрешение изображения, строгий парсер, проверка схемы и до трёх попыток с обратной связью остаются прежними. Меняется только текст промпта.

Главный критерий — F1@IoU 0.5, затем F1@IoU 0.75, затем меньше окончательных ошибок обработки. Время ниже — сумма обработки 100 изображений без ожидания GPU-блокировки и без загрузки модели. `parsing` и `validation` учитывают все отклонённые попытки; `failed` — изображения без допустимого ответа после повторов.

| Промпт | F1@0.5 | F1@0.75 | Precision@0.5 | Recall@0.5 | failed | parsing | validation | Попытки | Время, с |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [four_edges](05_four_edges/metrics.json) | 0.473684 | 0.070175 | 0.509434 | 0.442623 | 0 | 7 | 0 | 107 | 99.3 |
| [corners_then_sizes](04_corners_then_sizes/metrics.json) | 0.403226 | 0.064516 | 0.396825 | 0.409836 | 0 | 3 | 0 | 103 | 98.7 |
| [percentage_geometry](07_percentage_geometry/metrics.json) | 0.226415 | 0.000000 | 0.266667 | 0.196721 | 0 | 1 | 0 | 101 | 84.0 |
| [silent_checklist](10_silent_checklist/metrics.json) | 0.206186 | 0.000000 | 0.277778 | 0.163934 | 0 | 2 | 0 | 102 | 81.1 |
| [internal_1000](06_internal_1000/metrics.json) | 0.129870 | 0.000000 | 0.312500 | 0.081967 | 1 | 33 | 1 | 133 | 112.2 |
| [chinese](09_chinese/metrics.json) | 0.090909 | 0.000000 | 0.148148 | 0.065574 | 21 | 26 | 74 | 179 | 258.4 |
| [baseline](01_baseline/metrics.json) | 0.078740 | 0.000000 | 0.075758 | 0.081967 | 8 | 67 | 0 | 159 | 481.7 |
| [russian](08_russian/metrics.json) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 | 0 | 101 | 68.4 |
| [json_exemplars](03_json_exemplars/metrics.json) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 98 | 298 | 0 | 300 | 396.2 |
| [concise](02_concise/metrics.json) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 100 | 300 | 0 | 300 | 370.8 |

`concise`: 300 из 300 ответов начинались с Markdown code fence; окончательно отклонено 100 изображений. Парсер оставлен строгим: такие ответы не исправлялись автоматически.

`json_exemplars`: 298 из 300 ответов начинались с Markdown code fence; окончательно отклонено 98 изображений. Парсер оставлен строгим: такие ответы не исправлялись автоматически.

`russian`: 96/100 ответов содержали пустой boxes; failed=0, F1@0.5=0.000000. Допустимый JSON и геометрически возможные координаты сами по себе не означают правильное обнаружение объектов.

Лучший вариант: **four_edges**. F1@0.5 изменился с 0.078740 у повторённого baseline до 0.473684 (разница +0.394944). Точный текст сохранён в `best_prompt.txt` и `best_prompt.json`.

Исторический baseline: F1@0.5=0.078740, F1@0.75=0.000000, failed=8. Повторённый baseline: F1@0.5=0.078740, F1@0.75=0.000000, failed=8.

В каждом каталоге варианта сохранены config.yaml, predictions.jsonl, attempts.jsonl (все сырые ответы и причины отклонения), timings.jsonl, metrics.json, gpu.jsonl, environment.json и comet.json. GPU-сессии обеих моделей сериализует общий flock; расписание находится в ../gpu_schedule.jsonl. Новые веса и большие данные не скачивались. Comet ограничивал частоту загрузки артефактов: облачная копия может быть неполной; локальные файлы сохранены полностью, а SDK при необходимости сохраняет offline-архив для последующей синхронизации.

Это результаты подбора промпта на validation, а не независимая оценка качества. Выбор лучшего из десяти повышает риск подгонки к этим 100 изображениям; для оценки переноса нужна отдельная выборка. Основные конфиги модели не изменялись.

## Варианты

- `baseline`: Original detailed production prompt.
- `concise`: Short localization instruction without an example.
- `json_exemplars`: Explicit positive and negative valid JSON examples.
- `corners_then_sizes`: Internally locate opposite corners before conversion.
- `four_edges`: Identify separate outermost edges and avoid center/size confusion.
- `internal_1000`: Use Qwen's common 0–1000 grounding scale internally, then normalize.
- `percentage_geometry`: Percent-of-image mental geometry converted to fractions.
- `russian`: Russian task and geometry instructions with unchanged field names.
- `chinese`: Chinese localization and strict JSON instructions.
- `silent_checklist`: Internal identification, complete-object geometry, then syntax verification.
