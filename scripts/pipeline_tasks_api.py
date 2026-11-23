"""Resume one Yandex Tasks campaign using durable receipts for remote writes."""

import hashlib
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests


def write_file(path, content):
    path = Path(path)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TasksAPI:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        if session is None:
            self.session.headers["Authorization"] = f"OAuth {os.environ['YANDEX_TASKS_OAUTH_TOKEN']}"

    def request(self, method, path, *, params=None, body=None):
        response = self.session.request(
            method, f"https://tasks.yandex.ru/api/v1/{path}", params=params, json=body, timeout=60
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def items(self, path, **params):
        query = dict(params, sort="id", limit=50)
        while True:
            page = self.request("GET", path, params=query)
            yield from page["items"]
            if not page.get("has_more"):
                break
            last = page["items"][-1]["id"]
            if query.get("id_gt") == last:
                raise RuntimeError(f"Повторяющаяся страница API {path}")
            query["id_gt"] = last


class Receipts:
    """Idempotent restart after interruption, including a lost create/open response."""

    def __init__(self, path, api, poll):
        self.path, self.api, self.poll = path, api, poll
        self.state = json.loads(path.read_text()) if path.exists() else {}

    def save(self):
        write_file(self.path, json.dumps(self.state, ensure_ascii=False, indent=2))

    def resource(self, key, kind, payload):
        if kind in {"task-suites", "tasks"}:
            query = {"pool_id": payload["pool_id"]}

            def identity(item):
                tasks = item["tasks"] if kind == "task-suites" else [item]
                return [t["input_values"] for t in tasks]
        else:
            query = {"project_id": payload["project_id"]} if "project_id" in payload else {}
            field = {"projects": "private_comment", "skills": "name"}.get(kind, "private_name")

            def identity(item):
                return item.get(field)

        marker = identity(payload)
        fingerprint = digest(payload)
        record = self.state.get(key)
        if record is not None:
            if record["hash"] != fingerprint:
                raise ValueError(f"Входы этапа {key} изменились. Используйте отдельный каталог запуска.")
            if "id" in record:
                return record["id"]
            matches = [r for r in self.api.items(kind, **query) if identity(r) == marker]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Исход создания {key} неизвестен. Проверьте {kind} в кабинете. "
                    f"Запись находится в {self.path}; новый POST не отправлен."
                )
            created = matches[0]
        else:
            self.state[key] = {"hash": fingerprint}
            self.save()
            params = (
                {"open_pool": "false", "async_mode": "false"} if kind in {"tasks", "task-suites"} else None
            )
            try:
                created = self.api.request("POST", kind, body=payload, params=params)
            except requests.HTTPError as error:
                # Validation/auth/rate-limit failures did not create a resource. A timeout or conflict is ambiguous.
                if error.response.status_code in (400, 401, 403, 404, 422, 429):
                    del self.state[key]
                    self.save()
                raise
        self.state[key]["id"] = created["id"]
        self.save()
        return created["id"]

    def open(self, kind, resource_id):
        key = f"open/{kind}/{resource_id}"
        receipt = self.state.get(key)
        if receipt and receipt.get("done"):
            return
        if receipt is None:
            self.state[key] = {}
            self.save()
            try:
                op = self.api.request("POST", f"{kind}/{resource_id}/open")
            except requests.HTTPError as error:
                if error.response.status_code in (400, 401, 403, 404, 422, 429):
                    del self.state[key]
                    self.save()
                raise
            if "id" in op:
                self.state[key]["operation"] = op["id"]
                self.save()
        operation = self.state[key].get("operation")
        if operation:
            while True:
                op = self.api.request("GET", f"operations/{operation}")
                if op["status"] == "SUCCESS":
                    break
                if op["status"] in ("FAIL", "CANCELLED"):
                    del self.state[key]
                    self.save()
                    raise RuntimeError(f"Не удалось открыть {kind}/{resource_id}: {op.get('details')}")
                time.sleep(self.poll)
        else:
            pool = self.api.request("GET", f"{kind}/{resource_id}")
            if (
                pool.get("status") != "OPEN"
                and not pool.get("last_started")
                and not pool.get("last_close_reason")
            ):
                raise RuntimeError(
                    f"Проверьте состояние {kind}/{resource_id} после запроса на открытие. "
                    "Повторный запрос не отправлен."
                )
        self.state[key]["done"] = True
        self.save()
