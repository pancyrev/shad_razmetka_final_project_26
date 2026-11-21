"""Fresh annotations, two independent reviews, worker qualifications and final adjudication."""

import fcntl
import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from butterfly_bench.metrics import maximum_matches
from butterfly_bench.schema import validate_boxes, validate_photo_id
from scripts.pipeline_tasks_api import Receipts, TasksAPI, digest, write_file

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("FOUND", "NONE", "UNCERTAIN", "LOAD_ERROR")
VERDICTS = ("OK", "BAD", "LOAD_ERROR")
LANGUAGE = {"category": "profile", "key": "languages", "operator": "IN", "value": "RU"}
BROWSER = {"category": "computed", "key": "client_type", "operator": "EQ", "value": "BROWSER"}


def _project(review, marker):
    area = "review" if review else "annotation"
    interface = ROOT / "crowd" / area / "interface"

    def field(dtype, hidden=False, **values):
        return dict(type=dtype, hidden=hidden, required=True, **values)

    inputs = {"image": field("url"), "photo_id": field("integer", hidden=True)}
    scripts = ["$TOLOKA_ASSETS/js/toloka-handlebars-templates.js"]
    if review:
        title = "Проверка рамок вокруг бабочек"
        description = "Проверьте, верно ли выделены взрослые бабочки красными рамками."
        inputs.update(
            boxes_json=field("string"),
            worker_status=field("string", allowed_values=list(STATUSES)),
            source_assignment_id=field("string", hidden=True),
        )
        outputs = {"verdict": field("string", allowed_values=list(VERDICTS))}
    else:
        title = "Разметка бабочек на фотографии"
        description = "Нарисуйте прямоугольные рамки вокруг взрослых бабочек на фотографии."
        outputs = {
            "status": field("string", allowed_values=list(STATUSES)),
            "annotations": {"type": "json", "hidden": False, "required": False},
        }
        scripts.append("$TOLOKA_ASSETS/js/image-annotation.js")
    view = {
        "type": "classic",
        "assets": {"script_urls": scripts},
        "markup": (interface / "task.html").read_text(),
        "script": (interface / "task.js").read_text(),
        "styles": (interface / "task.css").read_text(),
        "settings": dict.fromkeys(
            (
                "showSkip",
                "showTimer",
                "showTitle",
                "showFinish",
                "showReward",
                "showSubmit",
                "showMessage",
                "showFullscreen",
                "showInstructions",
            ),
            True,
        ),
    }
    return {
        "public_name": title,
        "public_description": description,
        "private_comment": marker,
        "public_instructions": (interface / "instructions.html").read_text(),
        "assignments_issuing_type": "AUTOMATED",
        "assignments_automerge_enabled": False,
        "task_spec": {"input_spec": inputs, "output_spec": outputs, "view_spec": view},
    }


def _chunks(items, n):
    return [items[i : i + n] for i in range(0, len(items), n)]


def _photo(row):
    return {"input_values": {"image": row["photo_url"], "photo_id": row["photo_id"]}}


def _review_task(row):
    task = _photo(row)
    task["input_values"].update(
        worker_status=row["status"],
        source_assignment_id=row["source_assignment_id"],
        boxes_json=json.dumps(row["boxes"], separators=(",", ":")),
    )
    return task


def _control(row, correct, training=False):
    # Add an incorrect box to a verified negative, or omit a real object on a verified positive.
    boxes = row["boxes"]
    if not correct:
        boxes = boxes[1:] if boxes else [{"left": 0.1, "top": 0.2, "width": 0.5, "height": 0.4}]
    task = _review_task(
        dict(
            row,
            boxes=boxes,
            status="FOUND" if boxes else "NONE",
            source_assignment_id=f"control-{row['photo_id']}",
        )
    )
    verdict = "OK" if correct else "BAD"
    task["known_solutions"] = [{"output_values": {"verdict": verdict}, "correctness_weight": 1}]
    if training:
        task.update(
            infinite_overlap=True,
            message_on_unknown_solution=(
                "Рамки соответствуют видимым взрослым бабочкам."
                if correct
                else "Одна взрослая бабочка пропущена."
                if row["boxes"]
                else "На снимке без бабочек добавлена лишняя рамка."
            ),
        )
    return task


def _editor_box(shape):
    if "type" in shape and shape["type"] != "rectangle":
        raise ValueError("Ожидался прямоугольник.")
    if "data" not in shape:
        return {k: shape[k] for k in ("left", "top", "width", "height")}
    p, q = shape["data"]["p1"], shape["data"]["p2"]
    if any(type(point[axis]) not in (int, float) for point in (p, q) for axis in ("x", "y")):
        raise ValueError("Координаты должны быть числами.")
    left, right = sorted((p["x"], q["x"]))
    top, bottom = sorted((p["y"], q["y"]))
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def _answer(values, assignment, solution):
    status = solution.get("status")
    shapes = solution.get("annotations") or []
    try:
        if isinstance(shapes, str):
            shapes = json.loads(shapes)
        if isinstance(shapes, dict):
            shapes = shapes["shapes"]
        boxes = validate_boxes([_editor_box(shape) for shape in shapes])
        if status not in STATUSES or (status == "FOUND" and not boxes) or (status == "NONE" and boxes):
            raise ValueError("Несогласованные рамки и статус.")
    except (TypeError, KeyError, ValueError):
        # Keep the raw output in the export, and route unusable answers to correction/adjudication.
        status, boxes = "UNCERTAIN", []
    return {
        "photo_id": values["photo_id"],
        "photo_url": values["image"],
        "status": status,
        "boxes": boxes,
        "source_assignment_id": assignment["id"],
        "user_id": assignment["user_id"],
    }


def _write_rows(path, rows):
    write_file(path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows))


class CrowdRun:
    """Run with actual workers; receipts, raw assignment exports, votes and gold live in run_dir/crowd."""

    def __init__(self, run_dir, main_photos, reference_answers, poll_seconds=60, *, api=None):
        self.folder = Path(run_dir) / "crowd"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.main, self.reference = list(main_photos), list(reference_answers)
        main_ids = {validate_photo_id(r["photo_id"]) for r in self.main}
        ref_ids = {validate_photo_id(r["photo_id"]) for r in self.reference}
        if (
            not self.main
            or len(main_ids) != len(self.main)
            or len(ref_ids) != len(self.reference)
            or len(ref_ids) < 31
            or main_ids & ref_ids
        ):
            raise ValueError("Нужны уникальные рабочие фото и отдельные эталоны (минимум 31).")
        for r in self.reference:
            validate_boxes(r["boxes"])
        self.reference.sort(key=lambda row: row["photo_id"])
        self.api = api if api is not None else TasksAPI()
        self.poll = poll_seconds
        if self.poll < 0:
            raise ValueError("poll_seconds должен быть неотрицательным.")

    @contextmanager
    def _stage(self):
        with (self.folder / "run.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.ledger = Receipts(self.folder / "receipts.json", self.api, self.poll)
            meta = self.ledger.state.get("campaign")
            fingerprint = digest(
                [
                    [{k: r[k] for k in keys} for r in rows]
                    for rows, keys in (
                        (self.main, ("photo_id", "photo_url")),
                        (self.reference, ("photo_id", "photo_url", "boxes")),
                    )
                ]
            )
            if meta is None:
                now = datetime.now(UTC)
                meta = {
                    "id": uuid4().hex,
                    "history": str(uuid4()),
                    "fingerprint": fingerprint,
                    "expires": (now + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                self.ledger.state["campaign"] = meta
                self.ledger.save()
            elif meta["fingerprint"] != fingerprint:
                raise ValueError("Манифест или эталоны изменились после начала платного запуска.")
            self.meta = meta
            yield

    def _project(self, review):
        name = "review" if review else "annotation"
        return self.ledger.resource(name, "projects", _project(review, f"pipeline-{self.meta['id']}-{name}"))

    def _check_export(self, name, rows, *, reviews=False):
        saved = [json.loads(s) for s in (self.folder / f"{name}.jsonl").read_text().splitlines()]
        expected = {r["photo_id"]: r for r in saved}
        if reviews:
            for r in expected.values():
                del r["photo_id"]
        if rows != expected:
            raise ValueError(f"Ответы не соответствуют выгрузке {name} текущего запуска.")

    def _skill(self, name):
        title = f"BBox {name} {self.meta['id']}"
        return self.ledger.resource(
            "skill/" + name, "skills", {"name": title, "hidden": True, "public_name": {"RU": title}}
        )

    def _pool(self, name, project, reward, overlap, filters, qc=None, auto=True):
        payload = {
            "project_id": project,
            "private_name": f"{name} {self.meta['id']}",
            "reward_per_assignment": reward,
            "assignment_max_duration_seconds": 1200 if auto else 900,
            "auto_accept_solutions": auto,
            "auto_accept_period_day": 21 if auto else 14,
            "may_contain_adult_content": False,
            "will_expire": self.meta["expires"],
            "filter": {"and": [LANGUAGE, BROWSER, *filters]},
            "defaults": {"default_overlap_for_new_task_suites": overlap},
        }
        if qc:
            payload["quality_control"] = qc
        return self.ledger.resource("pool/" + name, "pools", payload)

    def _publish_pages(self, name, pool, pages, overlap):
        print(f"{name}: пул {pool}, страниц {len(pages)}, перекрытие {overlap}.", flush=True)
        for i, tasks in enumerate(pages):
            self.ledger.resource(
                f"{name}/page/{i}",
                "task-suites",
                {"pool_id": pool, "tasks": tasks, "overlap": overlap, "infinite_overlap": False},
            )
        self.ledger.open("pools", pool)

    def _assignments(self, name, pool, expected):
        path = self.folder / f"{name}-assignments.json"
        if path.is_file():
            return json.loads(path.read_text())
        last = None
        while True:
            records = list(self.api.items("assignments", pool_id=pool))
            valid = [r for r in records if r["status"] in ("SUBMITTED", "ACCEPTED")]
            if len(valid) == expected:
                write_file(path, json.dumps(records, ensure_ascii=False))
                return records
            if len(valid) > expected:
                raise ValueError(f"Лишние выдачи в {pool}: {len(valid)}/{expected}")
            if len(valid) != last:
                print(f"{name}: получено {len(valid)}/{expected} страниц.", flush=True)
                last = len(valid)
            status = self.api.request("GET", f"pools/{pool}")
            if status.get("status") in ("CLOSED", "ARCHIVED"):
                raise RuntimeError(
                    f"Пул {pool} закрыт с {len(valid)}/{expected} страницами. "
                    f"Причина: {status.get('last_close_reason')}. Проверьте его в кабинете."
                )
            time.sleep(self.poll)

    def _annotate_rows(self, name, pool, rows):
        pages = _chunks([_photo(r) for r in rows], 4)
        self._publish_pages(name, pool, pages, overlap=1)
        assignments = self._assignments(name, pool, len(pages))
        result = {}
        expected = {r["photo_id"]: r["photo_url"] for r in rows}
        for a in assignments:
            if a["status"] not in ("SUBMITTED", "ACCEPTED"):
                continue
            for task, answer in zip(a["tasks"], a["solutions"], strict=True):
                values = task["input_values"]
                row = _answer(values, a, answer["output_values"])
                pid = row["photo_id"]
                if expected.get(pid) != row["photo_url"] or pid in result:
                    raise ValueError(f"Чужой или повторный ответ разметки {pid} в пуле {pool}.")
                result[pid] = row
        if expected.keys() != result.keys():
            raise ValueError("В полученных ответах не хватает фото.")
        _write_rows(self.folder / f"{name}.jsonl", result.values())
        return result

    def annotate(self):
        """03a: qualify annotators on a real reference exam, then collect their first answers."""
        with self._stage():
            project = self._project(False)
            skill, participation = self._skill("annotation_exam"), self._skill("annotator")
            qc = {
                "configs": [
                    {
                        "collector_config": {"type": "ANSWER_COUNT", "parameters": {}},
                        "rules": [
                            {
                                "conditions": [
                                    {"key": "assignments_accepted_count", "operator": "GTE", "value": 1}
                                ],
                                "action": {
                                    "type": "RESTRICTION_V2",
                                    "parameters": {"scope": "POOL", "duration_unit": "PERMANENT"},
                                },
                            }
                        ],
                    }
                ]
            }
            pool = self._pool("annotation_exam", project, 2, 40, [], qc)
            exam = self.reference[:10]
            self._publish_pages("annotation_exam", pool, [[_photo(r) for r in exam]], 40)
            records = self._assignments("annotation_exam", pool, 40)
            truth = {r["photo_id"]: r["boxes"] for r in exam}
            scores = {}
            for a in records:
                user = a["user_id"]
                scores.setdefault(user, 0)
                if a["status"] not in ("ACCEPTED", "SUBMITTED"):
                    continue
                answers = [
                    _answer(t["input_values"], a, s["output_values"])
                    for t, s in zip(a["tasks"], a["solutions"], strict=True)
                ]
                if len(answers) != 10 or {r["photo_id"] for r in answers} != truth.keys():
                    raise ValueError(f"Неверный набор заданий в экзамене {a['id']}.")
                correct = sum(
                    r["status"] in ("FOUND", "NONE")
                    and len(r["boxes"])
                    == len(truth[r["photo_id"]])
                    == maximum_matches(r["boxes"], truth[r["photo_id"]], 0.5)
                    for r in answers
                )
                scores[user] = max(scores[user], 10 * correct)
            passed = {u: score for u, score in scores.items() if score > 80}
            if not passed:
                raise RuntimeError(f"Никто из {len(scores)} участников не набрал более 80% на экзамене.")
            if self.ledger.state.get("qualified_workers") != digest(scores):
                for user, score in scores.items():
                    self.api.request(
                        "PUT", "user-skills", body={"user_id": user, "skill_id": participation, "value": 100}
                    )
                    if user in passed:
                        self.api.request(
                            "PUT", "user-skills", body={"user_id": user, "skill_id": skill, "value": score}
                        )
                self.ledger.state["qualified_workers"] = digest(scores)
                self.ledger.save()
            filters = [{"category": "skill", "key": skill, "operator": "GTE", "value": 80}]
            work = self._pool("annotation_work", project, 2, 1, filters, auto=False)
            return self._annotate_rows("annotation_work", work, self.main)

    def _review_project(self):
        project = self._project(True)
        participant = self._skill("annotator")
        exam_skill, quality = self._skill("review_exam"), self._skill("review_quality")
        eligible = [{"category": "skill", "key": participant, "operator": "EQ"}]
        training = self.ledger.resource(
            "review_training",
            "trainings",
            {
                "project_id": project,
                "private_name": f"review_training {self.meta['id']}",
                "inherited_instructions": True,
                "may_contain_adult_content": False,
                "assignment_max_duration_seconds": 1200,
                "mix_tasks_in_creation_order": True,
                "shuffle_tasks_in_task_suite": False,
                "training_tasks_in_task_suite_count": 5,
                "task_suites_required_to_pass": 2,
                "retry_training_after_days": 1,
            },
        )
        for i, example in enumerate(self.reference[10:20]):
            task = _control(example, i % 2 == 0, training=True)
            self.ledger.resource(f"training/task/{i}", "tasks", dict(task, pool_id=training))
        self.ledger.open("trainings", training)

        def control_config(skills):
            rules = [
                {
                    "conditions": [{"key": "golden_set_answers_count", "operator": "GTE", "value": 10}],
                    "action": {
                        "type": "SET_SKILL_FROM_OUTPUT_FIELD",
                        "parameters": {"from_field": "golden_set_correct_answers_rate", "skill_id": s},
                    },
                }
                for s in skills
            ]
            return {
                "configs": [
                    {
                        "collector_config": {
                            "type": "GOLDEN_SET",
                            "uuid": self.meta["history"],
                            "parameters": {"history_size": 10},
                        },
                        "rules": rules,
                    }
                ]
            }

        exam_quality = control_config([exam_skill, quality])
        exam_quality["training_requirement"] = {
            "training_pool_id": training,
            "training_passing_skill_value": 80,
        }
        exam = self._pool("review_exam", project, 1, 30, eligible, exam_quality)
        tasks = [_control(r, i % 2 == 0) for i, r in enumerate(self.reference[20:30])]
        self._publish_pages("review_exam", exam, [tasks], 30)
        eligible += [
            {"category": "skill", "key": s, "operator": "GTE", "value": 80} for s in (exam_skill, quality)
        ]
        return project, eligible, control_config([quality])

    def _review(self, name, rows):
        project, filters, quality = self._review_project()
        pool = self._pool(name, project, 1, 3, filters, quality)
        pages = _chunks([_review_task(r) for r in rows.values()], 9)
        controls = self.reference[30:]
        for i, page in enumerate(pages):
            page.insert(i % (len(page) + 1), _control(controls[i % len(controls)], i % 2 == 0))
        self._publish_pages(name, pool, pages, 3)
        assignments = self._assignments(name, pool, 3 * len(pages))
        contributors = {r["user_id"] for r in rows.values()}
        votes = defaultdict(dict)
        for a in assignments:
            if a["status"] not in ("ACCEPTED", "SUBMITTED"):
                continue
            user = a["user_id"]
            if user in contributors:
                raise ValueError("Участник разметки оказался среди ревьюеров.")
            for t, sol in zip(a["tasks"], a["solutions"], strict=True):
                v = t["input_values"]
                if t.get("known_solutions") or v["source_assignment_id"].startswith("control-"):
                    continue
                pid = v["photo_id"]
                if (
                    pid not in rows
                    or v != _review_task(rows[pid])["input_values"]
                    or user in votes[pid]
                    or sol["output_values"]["verdict"] not in VERDICTS
                ):
                    raise ValueError(f"Некорректный или повторный голос по фото {pid}.")
                votes[pid][user] = sol["output_values"]["verdict"]
        if set(rows) != set(votes) or any(len(by_user) != 3 for by_user in votes.values()):
            raise ValueError("Не получены три независимых вердикта для каждого фото.")
        result = {}
        for pid, by_user in votes.items():
            tally = Counter(by_user.values())
            best, count = tally.most_common(1)[0]
            result[pid] = {
                "votes": dict(tally),
                "majority": best if count >= 2 else "TIE",
                "reviewers": sorted(by_user),
            }
        _write_rows(self.folder / f"{name}.jsonl", (dict(photo_id=i, **r) for i, r in result.items()))
        return result

    def review(self, first):
        """03b: training, exam, and three independent reviews of each new annotation."""
        with self._stage():
            self._check_export("annotation_work", first)
            return self._review("validation", first)

    def reannotate(self, first, votes, extra_ids=()):
        """03c: a new annotation for each non-OK, uncertain, failed, or explicitly flagged image."""
        with self._stage():
            self._check_export("annotation_work", first)
            self._check_export("validation", votes, reviews=True)
            extra_ids = set(extra_ids)
            if first.keys() != votes.keys() or not extra_ids <= first.keys():
                raise ValueError("Вердикты или дополнительные ID не соответствуют первому проходу.")
            changed = [
                r
                for pid, r in first.items()
                if pid in extra_ids
                or votes[pid]["majority"] != "OK"
                or r["status"] in ("LOAD_ERROR", "UNCERTAIN")
            ]
            fingerprint = digest(sorted(r["photo_id"] for r in changed))
            if self.ledger.state.setdefault("correction_selection", fingerprint) != fingerprint:
                raise ValueError("Состав переразметки изменился после начала этого этапа.")
            self.ledger.save()
            if not changed:
                _write_rows(self.folder / "reannotation.jsonl", [])
                return {}
            filters = [
                {"category": "skill", "key": self._skill("annotation_exam"), "operator": "GTE", "value": 80}
            ]
            pool = self._pool("reannotation", self._project(False), 2, 1, filters, auto=False)
            return self._annotate_rows("reannotation", pool, changed)

    def finish(self, first, votes, revised):
        """03d: independent re-review, wait for human adjudication of unresolved cases, export gold."""
        with self._stage():
            self._check_export("annotation_work", first)
            self._check_export("validation", votes, reviews=True)
            self._check_export("reannotation", revised)
            if first.keys() != votes.keys() or not revised.keys() <= first.keys():
                raise ValueError("Фотографии в первом и повторном проходах не совпадают с манифестом.")
            latest = first | revised
            if set(latest) != {r["photo_id"] for r in self.main}:
                raise ValueError("Неполный набор фото для итогового эталона.")
            reviews = votes | (self._review("revalidation", revised) if revised else {})
            disputed = {
                i: r
                for i, r in latest.items()
                if reviews[i]["majority"] != "OK" or r["status"] in ("UNCERTAIN", "LOAD_ERROR")
            }
            decisions = {}
            if disputed:
                path = self.folder / "adjudication.jsonl"
                if not path.exists():
                    _write_rows(
                        path,
                        (
                            {
                                "photo_id": i,
                                "photo_url": r["photo_url"],
                                "first_boxes": first[i]["boxes"],
                                "latest_boxes": r["boxes"],
                                "latest_status": r["status"],
                                "verdict": reviews[i]["majority"],
                                "final_boxes": None,
                            }
                            for i, r in disputed.items()
                        ),
                    )
                print(
                    f"Арбитраж {len(disputed)} снимков: проверьте фотографии и заполните final_boxes в {path}. "
                    "Используйте [] только для проверенного отрицательного примера.",
                    flush=True,
                )
                while True:
                    try:
                        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(self.poll)
                        continue
                    if (
                        {r["photo_id"] for r in entries} != disputed.keys()
                        or len(entries) != len(disputed)
                        or any(r["photo_url"] != disputed[r["photo_id"]]["photo_url"] for r in entries)
                    ):
                        raise ValueError(f"В {path} должны остаться все спорные ID с исходными ссылками.")
                    if all(r.get("final_boxes") is not None for r in entries):
                        decisions = {r["photo_id"]: validate_boxes(r["final_boxes"]) for r in entries}
                        break
                    time.sleep(self.poll)
            records, provenance = [], []
            for photo in self.main:
                i = photo["photo_id"]
                answer = latest[i]
                records.append(
                    {
                        "photo_id": i,
                        "photo_url": photo["photo_url"],
                        "boxes": decisions.get(i, answer["boxes"]),
                    }
                )
                source = "adjudicated" if i in decisions else "reannotation" if i in revised else "annotation"
                provenance.append(
                    {
                        "photo_id": i,
                        "source": source,
                        "status": answer["status"],
                        "review": reviews[i],
                        "source_assignment_id": answer["source_assignment_id"],
                    }
                )
            _write_rows(self.folder / "review.jsonl", provenance)
            path = self.folder / "golden.jsonl"
            _write_rows(path, records)
            return path
