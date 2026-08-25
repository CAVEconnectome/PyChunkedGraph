# pylint: disable=invalid-name, missing-docstring, broad-except
"""A TTL cache, in the pcg redis, for what find_path actually needs.

Measured on api6 aibs_v1dd root 864691132764660103 (240,928 level 2 ids):

    descend to the level 2 ids        5.5s
    read and reduce all cross edges  31.5s
    build the graph_tool graph        0.0s
    the shortest path itself          0.012s

The search is free; the cost is entirely data acquisition. Two namespaces cover it:

  * `descendants` -- the level 2 ids under one subtree, keyed by node id alone, since a
    node's children never change. This is what makes the 5.5s descent cheap.
  * `edges` -- the finished level 2 edge list of a whole root: 276,472 edges, 4.4MB. This
    is what makes the 31.5s read disappear.

An earlier version cached the raw per-subtree cross edge rows instead -- 65MB per object,
re-assembled into the same 4.4MB answer every call. It shipped 15x the bytes to recompute
something that had not changed, and a warm call still took 10.6s. Caching the answer
rather than its inputs is the point.

Root ids change on every edit, so an edge list is not looked up by the root being asked
about. It is found through that root's immediate predecessors
(`lineage.get_previous_root_ids`, a single read) and then patched: an edit replaces a
handful of level 2 nodes, so reading just their cross edges and splicing costs ~0.6s
against 31.5s to rebuild. Descendants are keyed by subtree id, which an edit leaves
untouched except along its own path.

The keys carry a TTL. The idle-time survey that motivated it found the mesh manifest keys
in this same instance sit at a median of 6 days idle, so under `allkeys-lru` these
displace stale bytes rather than anything live; `volatile-lru` would make that explicit.
"""

import logging
import os

import numpy as np

from ..utils import basetypes

# Same redis the mesh manifest cache uses (MANIFEST_CACHE_REDIS_* is what the helm chart
# wires into every pcg pod), under a separate key prefix.
REDIS_HOST = os.environ.get("MANIFEST_CACHE_REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("MANIFEST_CACHE_REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("MANIFEST_CACHE_REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

KEY_PREFIX = "pcgpath"

logger = logging.getLogger(__name__)

# The two namespaces expire on different clocks, because they cost and behave differently.
#
# An edge list is ~7.3MB and is superseded the moment its root is edited: the next call
# splices it forward under a new root id and never reads the old one again. A long TTL just
# accumulates dead entries -- a 20 edit session would leave ~147MB of them, only the newest
# of which is live. 15 minutes covers the gap between calls in a session while bounding it.
try:
    TTL_SECONDS = int(os.environ.get("PCG_PATH_CACHE_TTL_S", 900))
except ValueError:
    TTL_SECONDS = 900

# Descendants are the opposite: ~2MB for a whole object, keyed by subtree id, and an edit
# leaves all but a handful untouched, so one set serves every root a neuron passes through.
# They are also on the warm path -- the splice needs the new root's level 2 ids before it
# can diff -- so letting them expire costs the 5.5s descent even on an edge cache hit.
# Cheap and broadly reused, so they should outlive the edge lists rather than expire with
# them.
try:
    DESCENDANT_TTL_SECONDS = int(os.environ.get("PCG_PATH_CACHE_DESCENDANT_TTL_S", 3600))
except ValueError:
    DESCENDANT_TTL_SECONDS = 3600

# Keys per pipeline round trip. The cut is ~16k subtrees on a large object, so this is a
# handful of round trips rather than one enormous one.
try:
    PIPELINE_BATCH = int(os.environ.get("PCG_PATH_CACHE_BATCH", 5000))
except ValueError:
    PIPELINE_BATCH = 5000

_DISABLED = os.environ.get("PCG_PATH_CACHE_ENABLED", "1") == "0"




def _connect():
    """Connect, or return None. A missing redis must degrade to a plain cache miss."""
    if _DISABLED:
        return None
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
        client.ping()
        return redis.Redis.from_url(REDIS_URL)
    except Exception:
        return None


REDIS = _connect()


def _encode_graph(lvl2_ids, edges):
    """One blob holding both halves: a count, the level 2 ids, then the edge pairs."""
    lvl2_ids = np.ascontiguousarray(lvl2_ids, dtype=basetypes.NODE_ID)
    edges = np.ascontiguousarray(edges, dtype=basetypes.NODE_ID)
    return (
        np.array([len(lvl2_ids)], dtype=np.uint64).tobytes()
        + lvl2_ids.tobytes()
        + edges.tobytes()
    )


def _decode_graph(blob):
    """Inverse of _encode_graph, or None if the blob is not the shape we wrote."""
    try:
        n = int(np.frombuffer(blob, dtype=np.uint64, count=1)[0])
        body = np.frombuffer(blob, dtype=basetypes.NODE_ID, offset=8)
        if len(body) < n or (len(body) - n) % 2:
            return None
        return body[:n], body[n:].reshape(-1, 2)
    except Exception:
        return None


def _encode(array):
    """Pack an array of node ids. Fixed width, so decoding is one np.frombuffer view."""
    return np.ascontiguousarray(array, dtype=basetypes.NODE_ID).tobytes()


def _decode(blob, columns):
    """Inverse of _encode. Returns None if the blob is not a whole number of rows."""
    try:
        body = np.frombuffer(blob, dtype=basetypes.NODE_ID)
        if columns > 1 and len(body) % columns:
            return None
        return body.reshape(-1, columns)
    except Exception:
        return None


class SubtreeCache:
    """Subtree descendants and whole-root edge lists, for one graph.

    Separate namespaces because they are keyed differently and for different reasons:
    descendants by subtree id, which an edit leaves alone, and edge lists by root id,
    which an edit always replaces -- hence the predecessor lookup in `get_edges`.
    """

    def __init__(self, graph_id, client=REDIS):
        self._client = client
        self._graph_id = graph_id
        self.descendant_hits = 0
        self.descendant_misses = 0
        self.edge_hits = 0
        self.edge_misses = 0
        # Non-zero means a batch failed, so a low hit rate on this call is a redis problem
        # rather than a cold cache. Without this the two are indistinguishable.
        self.errors = 0

    @property
    def available(self):
        return self._client is not None

    def _descendant_key(self, node_id):
        return f"{KEY_PREFIX}:d:{self._graph_id}:{int(node_id)}"

    def _edge_key(self, root_id):
        return f"{KEY_PREFIX}:e:{self._graph_id}:{int(root_id)}"

    def _get_many(self, keys, node_ids, columns):
        """Pipelined MGET, decoding each blob as an (n, `columns`) array."""
        if self._client is None or not len(node_ids):
            return {}
        found = {}
        for start in range(0, len(node_ids), PIPELINE_BATCH):
            chunk_ids = node_ids[start : start + PIPELINE_BATCH]
            chunk_keys = keys[start : start + PIPELINE_BATCH]
            # Per batch, not around the whole loop. Wrapping the loop meant one failing
            # batch silently discarded every batch after it, so a partial read looked
            # exactly like a partial cache -- hit counts came back as suspiciously round
            # multiples of PIPELINE_BATCH with no way to tell the two apart.
            try:
                pipe = self._client.pipeline()
                for key in chunk_keys:
                    pipe.get(key)
                results = pipe.execute()
            except Exception as e:
                self.errors += 1
                # exc_info on the first only: enough to diagnose, not a flood.
                logger.warning(
                    "path cache read failed for %d keys: %s",
                    len(chunk_keys), e, exc_info=self.errors == 1,
                )
                continue
            for node_id, blob in zip(chunk_ids, results):
                if blob is None:
                    continue
                decoded = _decode(blob, columns)
                if decoded is not None:
                    found[int(node_id)] = decoded
        return found

    def _set_many(self, entries, key_of, ttl):
        if self._client is None or not entries:
            return
        items = list(entries.items())
        for start in range(0, len(items), PIPELINE_BATCH):
            try:
                pipe = self._client.pipeline()
                for node_id, value in items[start : start + PIPELINE_BATCH]:
                    pipe.setex(key_of(node_id), ttl, _encode(value))
                pipe.execute()
            except Exception as e:
                # Failing to populate must never fail the request, but it must be visible:
                # a write that half-succeeded is indistinguishable from a cold cache on
                # the next call, which is what made this hard to diagnose.
                self.errors += 1
                logger.warning(
                    "path cache write failed for %d keys: %s",
                    min(PIPELINE_BATCH, len(items) - start), e,
                    exc_info=self.errors == 1,
                )

    def get_descendants(self, node_ids):
        """{node_id: level 2 ids} for whatever is cached."""
        keys = [self._descendant_key(n) for n in node_ids]
        found = self._get_many(keys, node_ids, 1)
        found = {k: v.reshape(-1) for k, v in found.items()}
        self.descendant_hits += len(found)
        self.descendant_misses += len(node_ids) - len(found)
        return found

    def set_descendants(self, entries):
        self._set_many(entries, self._descendant_key, DESCENDANT_TTL_SECONDS)

    def get_edges(self, root_ids):
        """{root_id: (level 2 ids, edges)} for whichever roots are cached.

        Takes several roots because the caller looks up a root's *predecessors*, not the
        root itself: an edit mints a new root id, so the entry worth having is the one
        stored under the id this object carried a moment ago.
        """
        found = {}
        if self._client is None or not len(root_ids):
            self.edge_misses += len(root_ids)
            return found
        try:
            pipe = self._client.pipeline()
            for root_id in root_ids:
                pipe.get(self._edge_key(root_id))
            results = pipe.execute()
        except Exception as e:
            self.errors += 1
            logger.warning("path cache edge read failed: %s", e, exc_info=True)
            self.edge_misses += len(root_ids)
            return found
        for root_id, blob in zip(root_ids, results):
            if blob is None:
                continue
            decoded = _decode_graph(blob)
            if decoded is not None:
                found[int(root_id)] = decoded
        self.edge_hits += len(found)
        self.edge_misses += len(root_ids) - len(found)
        return found

    def set_edges(self, root_id, lvl2_ids, edges):
        if self._client is None:
            return
        try:
            self._client.setex(
                self._edge_key(root_id), TTL_SECONDS, _encode_graph(lvl2_ids, edges)
            )
        except Exception as e:
            self.errors += 1
            logger.warning("path cache edge write failed: %s", e, exc_info=True)
