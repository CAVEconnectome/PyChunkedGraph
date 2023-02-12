# pylint: disable=invalid-name, missing-docstring, too-many-arguments

from datastoreflex import DatastoreFlex

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

    def log_info(self, user_id, request_ts, response_time, path):
        self._add_log(
            log_type="INFO",
            user_id=user_id,
            path=path,
            request_ts=request_ts,
            response_time=response_time,
        )

    def log_error(self):
        ...

    def log_unhandled_exc(self):
        ...

    def _add_log(self, log_type, user_id, request_ts, response_time, path):
        key = self.client.key(self._kind, namespace=self._client.namespace)
        entity = self.client.entity(key)
        entity["type"] = log_type
        entity["user_id"] = user_id
        entity["date"] = request_ts
        entity["name"] = path
        entity["response_time(ms)"] = response_time
        self.client.put(entity)
        return entity.key.id
