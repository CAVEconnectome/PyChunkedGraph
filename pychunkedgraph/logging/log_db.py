# pylint: disable=invalid-name, missing-docstring, too-many-arguments

import os

from datastoreflex import DatastoreFlex
from flask import current_app


LOG_DB_CACHE = {}


class LogDB:
    def __init__(self, graph_id: str, client: DatastoreFlex):
        self._graph_id = graph_id
        self._client = client
        self._kind = f"server_logs_{self._graph_id}"

    @property
    def graph_id(self):
        return self._graph_id

    @property
    def client(self):
        return self._client

    def log_endpoint(self, user_id, request_ts, response_time, path):
        key = self.client.key(self._kind, namespace=self._client.namespace)
        entity = self.client.entity(key, exclude_from_indexes=("time_ms",))
        entity["user_id"] = user_id
        entity["date"] = request_ts
        entity["name"] = path
        entity["time_ms"] = response_time
        self.client.put(entity)
        return entity.key.id

    def log_function(self, operation_id, time_ms, path):
        key = self.client.key(self._kind, namespace=self._client.namespace)
        entity = self.client.entity(key, exclude_from_indexes=("time_ms",))
        entity["operation_id"] = operation_id
        entity["name"] = path
        entity["time_ms"] = time_ms
        self.client.put(entity)
        return entity.key.id

    def log_error(self):
        ...

    def log_unhandled_exc(self):
        ...


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
    return log_db
