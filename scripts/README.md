# Скрипты сбора и просмотра фотографий

Для установки изолированного окружения см. [инструкцию запуска](../docs/reproduce.md). Ниже команды выполняются из корня репозитория с Python этого окружения.

**Просмотр сохранённых данных.** Команды строят HTML по зафиксированным манифестам. Изображения в созданных галереях загружаются браузером с iNaturalist.

```bash
python scripts/prepare_review.py --csv outputs/galleries/selection.csv --gallery outputs/galleries/candidates.html
python scripts/build_annotation_gallery.py --urls-only --annotations data/contest_400/photos.jsonl --output outputs/galleries/contest.html
```

В [архиве отбора](../data/sampling/runs/20261108T001420_435829Z/README.md) сохранены кандидаты, назначения и контрольные суммы. `prepare_review.py` по умолчанию использует кандидатов этого запуска и [план](../data/sampling/sampling_plan.json). Для запуска по другим данным служат параметры `--candidates` и `--plan`.

**Восстановление оригиналов.** Подготовьте пустую папку внутри каталога скачиваний проекта и выполните в ней команды, указав абсолютный путь к репозиторию в `repo`. Загрузчик читает архивные назначения, проверяет формат, размеры, уникальность, записывает изображения в `images/`, а манифесты и журнал проверки — в текущую папку.

```bash
python "$repo/scripts/download_selected.py" --image-dir images
python "$repo/scripts/audit_similarity.py" --audit download_audit.jsonl --output similarity_review.csv
```

Для подтверждения побайтовой идентичности сверьте `sha256` для каждого `photo_id` с [историческим журналом](../data/sampling/runs/20261108T001420_435829Z/download_audit.jsonl). По доступным локальным оригиналам можно собрать галерею с наложенными рамками, указав `--images` и `--output`.

```bash
python "$repo/scripts/build_annotation_gallery.py" --annotations "$repo/data/reference_100/annotations.jsonl" --images images --output reference_100.html
```

**Новая выборка.** `collect_candidates.py` обращается к API iNaturalist. Флаг `--dry-run` показывает запросы без отправки. Сохранённые ответы API и фиксированный seed позволяют воспроизвести исторический отбор. Новые ответы API зависят от даты запроса.
