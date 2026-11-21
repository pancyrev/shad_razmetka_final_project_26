# Воспроизведение проекта

Все команды запускаются из корня репозитория.

## Повторить исходный эксперимент

Подготовьте окружение и CUDA-сборку llama.cpp по разделу «Инференс на NVIDIA GPU» ниже. Затем выполните `python -m jupyterlab pipeline.ipynb` и выберите **Run All Cells**. [Основной ноутбук](../pipeline.ipynb)

По умолчанию `MODE = "historical"`. Исходные данные — снимок 700 кандидатов iNaturalist, конкурсный список и ответы людей. Ноутбук заново выбирает 400 основных и 100 вспомогательных фото с seed `20261107`, скачивает оригиналы и проверяет их SHA-256. Из первых 400 ответов и 43 исправлений строится эталон. Перед инференсом проверяются порядок фотографий, рамки и пиксели относительно исходного Parquet. Все четыре модели запускаются с seed `42`, исходными ревизиями весов и параметрами. В конце сравниваются свежие предсказания и F1 с опубликованными. На другой версии CUDA, железа или библиотек возможны расхождения. [Снимок кандидатов](../data/sampling/runs/20261108T001420_435829Z/) · [Исходные результаты моделей](../results/final_400/)

Каждый новый прогон пишет результаты в `outputs/pipeline/<RUN_ID>/`, оригиналы — в `../downloads/final-project/pipeline/<RUN_ID>/images/`. Дополнительно загружаются исходный Parquet (около 517 МБ) и веса моделей. Идентификатор выводится в первой части ноутбука. Для продолжения прерванного запуска впишите его в `RESUME_RUN`. После инференса сравнение моделей лежит в `comparison.jsonl`, отдельные предсказания и журналы — в подкаталоге `models/` того же прогона.

## Собрать новые ответы людей

Режим `MODE = "live"` обращается к сегодняшнему API iNaturalist. Он использует те же квоты и seed, но выдача API и ответы людей меняются. Заказчик размечает новую вспомогательную сотню в Jupyter, а затем ячейки создают и открывают **платные** пулы через `YANDEX_TASKS_OAUTH_TOKEN`. Ядро ждёт исполнителей и трёх независимых вердиктов. При спорном результате повторной проверки потребуется заполнить `crowd/adjudication.jsonl` в папке текущего запуска. [Правила и исходные пулы](annotation.md)

## Посмотреть сохранённый прогон на CPU

Для пересчёта метрик по готовым предсказаниям достаточно Python 3.12 и окружения с Jupyter.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,crowd,notebook]'
python -m jupyterlab notebooks/reproduce_saved_results.ipynb
```

В этом ноутбуке сохранены таблицы и изображения предыдущего прогона. При **Run All Cells** метрики по умолчанию пересчитываются локально. [Ноутбук для CPU](../notebooks/reproduce_saved_results.ipynb) · [Версии библиотек исходного отбора](../requirements/data-collection.lock.txt)

Для тестов установите группу `dev` в том же окружении: `python -m pip install -e '.[dev]'`. Проверенный состав локального окружения приведён в [CPU lock-файле](../requirements/local-macos-py312.lock.txt). Для его восстановления на macOS ARM64 с Python 3.12 выполните `python -m pip install -r requirements/local-macos-py312.lock.txt`, затем `python -m pip install --no-deps -e .`. Для удаления созданного окружения завершите его командой `deactivate` и удалите только каталог `.venv/`. Данные и результаты находятся отдельно.

## Инференс на NVIDIA GPU

Пайплайн запускает модели последовательно на CUDA. OWLv2 и Grounding DINO
используют Transformers и FP32, Qwen3-VL — Transformers и BF16 без квантования,
Gemma 4 — CUDA-сервер llama.cpp с UD-Q4_K_XL и визуальным проектором F16.
Конфигурации находятся в `butterfly_bench/conf/`. Ноутбук
[демонстрации моделей](../notebooks/models/03_model_demo.ipynb) использует тот же код, что и CLI.

### Установка окружения для инференса

На машине нужны Python 3.12+, драйвер NVIDIA, а для Gemma ещё CUDA Toolkit,
Git и совместимый C++-компилятор. Исходный прогон выполнен на RTX 3080 с 10 ГиБ памяти.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[inference,dev,notebook]'
bash scripts/setup_llama_cpp.sh
```

Скрипт сборки фиксирует commit llama.cpp из `scripts/llama_cpp_revision.txt`,
использует CUDA sm_86 и GCC 13 для установленного CUDA Toolkit 12.4.
Для другой машины можно задать `CC`, `CXX`, `CUDAHOSTCXX`, `CUDA_ARCHITECTURES`
и `BUILD_JOBS`. Исходники и сборка в `tools/` не входят в Git.
Зависимости проверенного окружения записаны в `requirements/inference-cu128.lock.txt`.
Для его восстановления сначала установите PyTorch из CUDA-индекса, затем:

```bash
python -m pip install -r requirements/inference-cu128.lock.txt
python -m pip install -e . --no-deps
python -m pip install -c requirements/inference-cu128.lock.txt 'jupyterlab>=4,<5'
```

На ноутбуке проекта кеш для загрузок можно задать перед запуском: `export HF_HOME="$PWD/../downloads/final-project/huggingface"`.
Веса автоматически скачиваются в HF-кеш: только нужные файлы,
в том числе отдельно `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` и `mmproj-F16.gguf`.
Публичные модели и датасет читаются без токена. Веса и датасет не загружаются в Git.

## Comet

Задайте `COMET_API_KEY` через окружение, локальный `.env` или `~/.comet.config`.
Ключи не записываются в конфигурацию и артефакты запуска.
`comet.mode=auto` выбирает онлайн-эксперимент при наличии ключа,
иначе создаёт настоящий offline-эксперимент Comet в каталоге результатов.
Можно явно выбрать `online`, `offline` или `disabled`.

```bash
python -m butterfly_bench model=owlv2 comet.project_name=butterfly-bench
python -m butterfly_bench model=qwen comet.mode=offline data.limit=4
```

Параметры, метрики, версии, предсказания и журнал ответов логируются явно
через Comet SDK: интеграция `Trainer` здесь не нужна, поскольку обучения нет.
`comet.log_images=4` дополнительно отправляет первые четыре изображения с рамками;
по умолчанию фотографии не отправляются. `comet.log_raw_responses=false`
отключает отправку сырых ответов, сохраняя их локально.
В offline-режиме ничего не отправляется. Локальные результаты сохраняются
при всех режимах Comet.

## Тестирование по эталонам

В `xartik/inaturalist_butterflies` используются новые имена split:

| Split | Состав | Назначение |
| --- | --- | --- |
| `train` | 100 изображений, 61 рамка, 42 отрицательных примера | Подбор промптов; прежний `validation` |
| `validation` | 400 изображений, 241 рамка, 180 отрицательных примеров | Оценка моделей; прежний `test` |
| `test` | 400 конкурсных изображений без разметки | Отдельная конкурсная выборка |

По умолчанию выбран `validation` (400 размеченных изображений).
Имя split не меняет режим: `mode=test` означает оценивание по известным рамкам,
а не выбор HF split `test`. Данные без эталонов требуют `mode=inference`;
при попытке оценивания загрузчик сообщает об отсутствии разметки.

Проверка переименования, загрузки всех 500 размеченных изображений, четырёх
CUDA-моделей и демо — [в отчёте](../results/experiments/split_migration_2026-11-16/README.md).

```bash
# Быстрая проверка на train
python -m butterfly_bench model=owlv2 data.split=train data.limit=8
python -m butterfly_bench model=grounding_dino data.split=train data.limit=8
python -m butterfly_bench model=qwen_tuned data.split=train data.limit=8
python -m butterfly_bench model=gemma_tuned data.split=train data.limit=8

# Все четыре модели на одинаковой полной валидации, последовательно
run_dir=$(mktemp -d "$PWD/outputs-replay-XXXXXX")
for model in owlv2 grounding_dino qwen gemma; do
  python -m butterfly_bench \
    --config-path "$PWD/results/final_400/frozen" --config-name "$model" \
    "hydra.run.dir=$run_dir/$model" 'output_dir=${hydra:runtime.output_dir}' comet.mode=disabled
done

# Оценка на всех 400 размеченных validation после выбора настроек на train
python -m butterfly_bench model=grounding_dino data.split=validation

# Собственная эталонная JSONL-разметка с image_path или photo_url
python -m butterfly_bench model=owlv2 data=manifest \
  data.path='data/reference_100/annotations.jsonl' \
  data.image_root=../downloads/final-project/images data.cache_dir=../downloads/final-project/manifest-cache
```

При выборе поднабора `data.split=train data.photo_ids=[755630032,760199895]` проверьте, что ID
есть в выбранном split. Можно использовать `data.shuffle=true data.limit=12`;
перемешивание зависит только от `seed`, а не от модели. `data.limit=null`
обрабатывает весь выбранный набор. Одни и те же настройки данных дают одинаковые ID.
Категории наблюдений и эталонные рамки не передаются моделям.

## Инференс без эталонов

```bash
python -m butterfly_bench mode=inference model=grounding_dino \
  data=manifest data.path=data/contest_400/photos.jsonl \
  data.image_root=../downloads/final-project/images data.cache_dir=../downloads/final-project/manifest-cache
```

Входной JSONL содержит целочисленный `photo_id` и `image_path` либо `photo_url`.
Относительный `image_path` считается от каталога манифеста. Без `image_path`
сначала проверяется `data.image_root/<photo_id>.jpg`, затем скачивается `photo_url`
в каталог `data.cache_dir`. В командах выше для загрузок выделена папка `../downloads/final-project/manifest-cache/`. Ошибка открытия или загрузки изображения фиксируется как ошибка,
а не как отсутствие бабочек. Режим `inference` игнорирует эталоны, даже если они есть.

## Промпты, повторные попытки и рамки

У каждой модели есть `model.prompt`. Стартовый запрос детекторов — `butterfly`;
для DINO автоматически добавляется завершающая точка. Qwen и Gemma наследуют
одинаковый подробный промпт, который описывает правила выделения и точную схему JSONL.
Промпт можно заменить через CLI или YAML. `photo_id` добавляется автоматически.

```bash
python -m butterfly_bench model=owlv2 'model.prompt=an adult butterfly' model.threshold=0.15
python -m butterfly_bench model=qwen retries.max_attempts=4
python -m butterfly_bench model=gemma model.context_size=4096
```

Парсер VLM принимает строго одну JSONL-строку без Markdown, комментариев и
дублирующихся ключей. У записи ровно три поля: `photo_id`, `boxes`, `error`.
У рамки ровно `left`, `top`, `width`, `height`. Проверяются ID, типы,
конечность чисел, положительные размеры и границы изображения.
Строки вместо чисел, `NaN`, пиксельные координаты, неверные ID и неизвестные поля
отклоняются. Разрешена лишь погрешность сложения float до 1e-12 на границе.
JSON автоматически не извлекается из произвольного текста и не «чинится».

По умолчанию выполняется не более трёх попыток **всего**. При отказе парсера
или валидатора VLM получает причину ошибки и то же изображение.
Первая генерация жадная; повторные используют `retry_temperature=0.2`
и воспроизводимые разные seed. Детекторы проходят проверку координат и scores,
затем явное обрезание рамок по границам; число обрезанных рамок логируется.
Полностью вырожденные рамки и неконечные значения считаются ошибками.
Ни одна структурная проверка не подтверждает, что объект действительно бабочка.

NMS по умолчанию отключён. Для дополнительного эксперимента можно изменить
только этот фактор и сравнить результаты на тех же данных:

```bash
python -m butterfly_bench -m model=owlv2 postprocess.nms_iou=null,0.5
```

NMS упорядочивает рамки по score детектора. Для VLM
используется одинаковый служебный вес 1.0, поэтому сохраняется порядок
их выдачи. Эти веса в `timings.jsonl` не являются уверенностью VLM.

Для Gemma зафиксированы `image_max_tokens=560`, `ubatch_size=1024`,
`batch_size=1024` и уменьшение длинной стороны до 1280 пикселей.
Бюджет визуальных токенов задан явно: у этой модели llama.cpp требует, чтобы
изображение помещалось в один ubatch. Сервер запускается с `--fit off`,
а пайплайн проверяет подтверждение полного offload слоёв в журнале;
неподтверждённый запуск не принимается за CUDA-инференс.

## Метрики и артефакты

Для каждого IoU внутри изображения находится **максимальное число** пар рамок;
допускается переназначение пар, каждая рамка используется не более одного раза.
Это не жадное сопоставление. TP/FP/FN суммируются по набору, затем считаются
micro precision, recall и F1. Пороги по умолчанию 0.5 и 0.75.
Основные `precision`, `recall`, `f1`, `errors` относятся к `primary_iou=0.5`.
`errors=FP+FN`; технические сбои отражаются отдельно в `processing_errors`.
Также сохраняются число изображений с ошибками и FP на изображениях без бабочек.
Нулевой знаменатель precision/recall/F1 даёт 0.

Неудачно обработанные фотографии **не исключаются** из оценки: их эталонные
объекты дают FN, а технический сбой остаётся в отдельном счётчике.

Каждый запуск создаёт новый каталог в `outputs/` (или `multirun/`):

| Файл | Содержимое |
| --- | --- |
| `config.yaml` | Разрешённый Hydra-конфиг, промпт, фактические HF commit весов и датасета |
| `data.json` | Источник, split, точный упорядоченный список ID, хеш локального манифеста |
| `environment.json`, `model.json` | Версии библиотек, Git, CUDA, GPU, dtype и версия llama.cpp |
| `predictions.jsonl` | Одна запись на ID; при окончательной ошибке `boxes=null`, `error` содержит причину |
| `attempts.jsonl` | Исходные ответы, координаты, ошибки парсинга/валидации/исполнения и seed попыток |
| `timings.jsonl` | Время и число попыток на изображение, scores после обработки |
| `metrics.json` | Метрики качества, времени и использования GPU |
| `gpu.jsonl` | Секундные замеры общей памяти и загрузки устройства через nvidia-smi |
| `comet.json` | Режим, ID и URL Comet-эксперимента |
| `llama-server.log` | Размещение слоёв, проектор, CUDA-буферы и диагностика Gemma |
| `status.json` | `running`, `complete`, `completed_with_errors` или `failed` |
| `submission.jsonl` | Создаётся только при успешной обработке **всех выбранных** ID |

Время `processing_seconds` включает открытие изображений, инференс, проверки
и все повторные попытки; первая CUDA-итерация тоже включена. Подготовка датасета
и загрузка модели указаны отдельно. Это последовательная обработка по одному
изображению, без отдельного прогрева. `device_peak_used_mib` включает другие
процессы на устройстве; `torch_peak_allocated_mib` относится к текущему PyTorch,
поэтому не измеряет память отдельного процесса llama-server.

Одиночный запуск CLI возвращает код 2 при окончательных ошибках изображений.
В Hydra multirun такие ошибки записываются в статус и журнал, а оставшиеся
модели продолжают запускаться; проверяйте `processing_errors` каждой модели.
Не отправляйте
диагностический `predictions.jsonl` со сбоями в Контест: исправьте причину
и повторите запуск. Для Контеста выберите полный манифест из 400 изображений;
файл для поднабора из демо не является полной посылкой. Пайплайн ничего
не загружает в Контест или GitHub.

`seed=42` фиксирует генераторы Python/NumPy/PyTorch, отключает TF32 и включает
детерминированные операции PyTorch. Seed каждой попытки зависит от ID,
а не от порядка модели/изображения. llama.cpp получает тот же seed.
Побитовое совпадение между разными GPU, сборками llama.cpp и версиями библиотек
не гарантируется; для воспроизведения используйте сохранённые версии и commit.

## Проверки реализации

Фактические результаты четырёх моделей на всех 100 изображениях нынешнего train
и эксперимент с NMS сохранены в
[отчёте](../results/experiments/validation_2026-11-13/README.md).

```bash
python -m pytest -q
ruff check butterfly_bench tests
```

Тесты покрывают невозможные координаты и scores, строгий JSONL, обе стадии
отказа VLM и повторные попытки, технические ошибки вместо пустых ответов,
отрицательные изображения, повторные рамки и пример, где жадное сопоставление
даёт неправильную F1.

## Промпты после подбора на train

В архивных отчётах сохранены имена split и ревизии на момент запуска:
старый validation соответствует нынешнему train, старый test — нынешнему validation.
Сырые ответы, конфиги и метрики прошлых экспериментов не переименовываются.

Формат с `photo_id` в запросе и полном ответе модели восстановлен после
эксперимента без служебных полей. Повторная проверка выбранных промптов —
[в отдельном отчёте](../results/experiments/restored_task_record_2026-11-16/README.md).

По десять промптов Qwen и Gemma проверены на одинаковых 100 изображениях нынешнего train. Точные варианты, метрики и ограничения сравнения — [в отчёте](../results/experiments/prompt_search_2026-11-15/README.md). Исходные конфигурации сохранены; лучшие испытанные запросы доступны как `model=qwen_tuned` и `model=gemma_tuned`. У пресетов зафиксирована ревизия весов, остальные настройки наследуются от соответствующей исходной модели.

```bash
python -m butterfly_bench model=qwen_tuned data.split=train
python -m butterfly_bench model=gemma_tuned data.split=train
```

Запускайте эти команды последовательно. В демо-ноутбуке также можно выбрать один из этих пресетов. Это результаты адаптивного подбора на train; отдельная размеченная validation при подборе не использовалась.

Скрипты `scripts/prompt_search_qwen.py` и `scripts/prompt_search_gemma.py` теперь
используют только train (100 изображений), с закреплённой новой ревизией данных.
Новые результаты пишутся в `outputs/prompt_search_train`; путь можно задать
через `BUTTERFLY_PROMPT_SEARCH_RESULTS`. Архивный подбор не перезаписывается.

Итоговые четыре конфигурации проверены на всех 400 изображениях нынешнего HF split `validation`
(в исходной ревизии он назывался `test`):
[метрики, полные конфиги и Comet-эксперименты](../results/final_400/README.md).
OWLv2 оценивался с выбранным на нынешнем train NMS=0.5; остальные — без NMS.
