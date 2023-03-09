# pylint: disable=invalid-name, missing-docstring, import-outside-toplevel, broad-exception-caught

import os
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import List

import redis
import numpy as np

DOES_NOT_EXIST = "X"

REDIS_HOST = os.environ.get("MANIFEST_CACHE_REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("MANIFEST_CACHE_REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("MANIFEST_CACHE_REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


REDIS = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
try:
    REDIS.ping()
    REDIS = redis.Redis.from_url(REDIS_URL)
except Exception:
    REDIS = None


class ManifestCache:
    def __init__(self, namespace: str, initial: Optional[bool] = True) -> None:
        self._initial = initial
        self._namespace = namespace

    @property
    def initial(self) -> str:
        return self._initial

    @property
    def namespace(self) -> str:
        return self._namespace

    def _get_cached_initial_fragments(self, node_ids: List[np.uint64]):
        if REDIS is None:
            return {}, node_ids

        pipeline = REDIS.pipeline()
        for node_id in node_ids:
            pipeline.get(f"{self.namespace}:{node_id}")

        result = {}
        not_cached = []
        not_existing = []
        fragments = pipeline.execute()
        for node_id, fragment in zip(node_ids, fragments):
            if fragment is None:
                not_cached.append(node_id)
                continue
            fragment = fragment.decode()
            try:
                path, offset, size = fragment.split(":")
            except ValueError:
                not_existing.append(node_id)
            result[node_id] = [path, int(offset), int(size)]
        return result, not_cached, not_existing

    def _get_cached_dynamic_fragments(self, node_ids: List[np.uint64]):
        if REDIS is None:
            return {}, node_ids

        pipeline = REDIS.pipeline()
        for node_id in node_ids:
            pipeline.get(f"{self.namespace}:{node_id}")

        result = {}
        not_cached = []
        not_existing = []
        fragments = pipeline.execute()
        for node_id, fragment in zip(node_ids, fragments):
            if fragment is None:
                not_cached.append(node_id)
                continue
            fragment = fragment.decode()
            if fragment == DOES_NOT_EXIST:
                not_existing.append(node_id)
            else:
                result[node_id] = fragment
        return result, not_cached, not_existing

    def get_fragments(self, node_ids) -> Tuple[Dict, List[np.uint64], List[np.uint64]]:
        if self.initial is True:
            return self._get_cached_initial_fragments(node_ids)
        return self._get_cached_dynamic_fragments(node_ids)

    def _set_cached_initial_fragments(
        self, fragments_d: Dict, not_existing: List[np.uint64]
    ) -> None:
        if REDIS is None:
            return

        pipeline = REDIS.pipeline()
        for node_id, fragment in fragments_d.items():
            path, offset, size = fragment
            pipeline.set(f"{self.namespace}:{node_id}", f"{path}:{offset}:{size}")

        for node_id in not_existing:
            pipeline.set(f"{self.namespace}:{node_id}", DOES_NOT_EXIST)

        pipeline.execute()

    def _set_cached_dynamic_fragments(
        self, fragments_d: Dict, not_existing: List[np.uint64]
    ) -> None:
        if REDIS is None:
            return

        pipeline = REDIS.pipeline()
        for node_id, fragment in fragments_d.items():
            pipeline.set(f"{self.namespace}:{node_id}", fragment)

        for node_id in not_existing:
            pipeline.set(f"{self.namespace}:{node_id}", DOES_NOT_EXIST)

        pipeline.execute()

    def set_fragments(self, fragments_d: Dict, not_existing: List[np.uint64]):
        if self.initial is True:
            self._set_cached_initial_fragments(fragments_d, not_existing)
        else:
            self._set_cached_dynamic_fragments(fragments_d, not_existing)
