# pylint: disable=invalid-name, missing-docstring, too-many-arguments

import os
import threading
import time
import queue
from functools import wraps

from datastoreflex import DatastoreFlex
from flask import current_app


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

    def log_endpoint(self, path: str, user_id, request_ts, response_time):
        item = {
            "name": path,
            "user_id": int(user_id),
            "request_ts": request_ts,
            "time_ms": response_time,
        }
        self._q.put(item)

    def log_function(self, name: str, operation_id, time_ms):
        item = {"name": name, "operation_id": int(operation_id), "time_ms": time_ms}
        self._q.put(item)

    def log_entity(self):
        while True:
            try:
                item = self._q.get_nowait()
                key = self.client.key(self._kind, namespace=self._client.namespace)
                entity = self.client.entity(key, exclude_from_indexes=("time_ms",))
                entity.update(item)
                self.client.put(entity)
            except queue.Empty:
                time.sleep(1)


def get_log_db(graph_id: str) -> LogDB:
    try:
        return LOG_DB_CACHE[graph_id]
    except KeyError:
        ...

    namespace = os.environ.get("PCG_SERVER_LOGS_NS", "pcg_server_logs_test")
    client = DatastoreFlex(
        project=current_app.config["PROJECT_ID"], namespace=namespace
    )

    log_db = LogDB(graph_id, client=client)
    LOG_DB_CACHE[graph_id] = log_db
    threading.Thread(target=log_db.log_entity, daemon=True).start()
    return log_db


def log_metrics(func, *, graph_id: str, operation_id: int = None):
    """
    Decorator to log metrics of a function to LogDB.
    Currently only logs time elapsed(ms).
    """

    @wraps(func)
    def log_time(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        time_ms = (time.time() - start) * 1000
        log_db = get_log_db(graph_id)
        log_db.log_function(
            name=func.__name__,
            operation_id=operation_id,
            time_ms=time_ms,
        )
        return result
    return log_time
