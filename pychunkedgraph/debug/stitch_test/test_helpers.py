"""Shared test utilities for stitch_test."""

import numpy as np

from pychunkedgraph.graph import basetypes

from .resolver import resolve_svs_to_layer
from .wave_cache import WaveCache

NODE_ID = basetypes.NODE_ID


def noop_read(ids):
    pass


def make_cache(**kw) -> WaveCache:
    c = WaveCache(noop_read)
    c.begin_stitch()
    c.old_to_new = kw.get("old_to_new", {})
    c.unresolved_acx = kw.get("unresolved_acx", {})
    return c


def get_parent(c: WaveCache, nid: int) -> int:
    return int(c.get_parents(np.array([nid], dtype=NODE_ID))[0])


def get_children(c: WaveCache, nid: int):
    return c.get_children_batch(np.array([nid], dtype=NODE_ID)).get(int(nid), np.array([], dtype=NODE_ID))


def get_acx(c: WaveCache, nid: int) -> dict:
    return c.get_acx_batch(np.array([nid], dtype=NODE_ID)).get(int(nid), {})


def get_cx(c: WaveCache, nid: int) -> dict:
    return c.get_cx_batch(np.array([nid], dtype=NODE_ID)).get(int(nid), {})


def resolve_sv(sv: int, target_layer: int, cache, get_layer) -> int:
    """Resolve a single SV to target_layer. Wrapper around batch resolver."""
    result = resolve_svs_to_layer(np.array([sv], dtype=NODE_ID), target_layer, cache, get_layer)
    return result.get(sv, sv)
