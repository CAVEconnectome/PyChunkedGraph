# pylint: disable=invalid-name, missing-docstring, too-many-arguments

import os
import threading
import time
import queue
from datetime import datetime

from google.api_core.exceptions import GoogleAPIError
from datastoreflex import DatastoreFlex


LOG_DB_CACHE = {}


class LogDB:
    def __init__(self, graph_id: str, client: DatastoreFlex):
        self._graph_id = graph_id
        self._client = client
        self._kind = f"server_logs_{self._graph_id}"
        self._q = queue.Queue()

    @property
    def graph_id(self):
        return self._graph_id

    @property
    def client(self):
        return self._client

    def log_endpoint(
        self,
        path,
        args,
        user_id,
        operation_id,
        request_ts,
        response_time,
    ):
        item = {
            "name": path,
            "args": args,
            "user_id": str(user_id),
            "request_ts": request_ts,
            "time_ms": response_time,
        }
        if operation_id is not None:
            item["operation_id"] = int(operation_id)
        self._q.put(item)

    def log_code_block(self, name: str, operation_id, timestamp, time_ms):
        item = {
            "name": name,
            "operation_id": int(operation_id),
            "request_ts": timestamp,
            "time_ms": time_ms,
        }
        self._q.put(item)

    def log_entity(self):
        while True:
            try:
                item = self._q.get_nowait()
                key = self.client.key(self._kind, namespace=self._client.namespace)
                entity = self.client.entity(
                    key,
                    exclude_from_indexes=(
                        "args",
                        "time_ms",
                        "user_id",
                    ),
                )
                entity.update(item)
                self.client.put(entity)
            except queue.Empty:
                time.sleep(1)


def get_log_db(graph_id: str) -> LogDB:
    try:
        return LOG_DB_CACHE[graph_id]
    except KeyError:
        ...

    try:
        project = os.environ["PCG_SERVER_LOGS_PROJECT"]
    except KeyError as err:
        raise GoogleAPIError(f"Datastore project env not set: {err}") from err

    namespace = os.environ.get("PCG_SERVER_LOGS_NS", "pcg_server_logs_test")
    client = DatastoreFlex(project=project, namespace=namespace)

    log_db = LogDB(graph_id, client=client)
    LOG_DB_CACHE[graph_id] = log_db
    # use threads to exclude time reguired to log
    threading.Thread(target=log_db.log_entity, daemon=True).start()
    return log_db


class TimeIt:
    def __init__(self, name: str, graph_id: str, operation_id):
        self._name = name
        self._start = None
        self._graph_id = graph_id
        self._operation_id = int(operation_id)
        self._ts = datetime.utcnow()

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        time_ms = (time.time() - self._start) * 1000
        try:
            log_db = get_log_db(self._graph_id)
            log_db.log_code_block(
                name=self._name,
                operation_id=self._operation_id,
                timestamp=self._ts,
                time_ms=time_ms,
            )
        except GoogleAPIError:
            ...
