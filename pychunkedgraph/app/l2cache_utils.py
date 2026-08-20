# pylint: disable=invalid-name, missing-docstring, line-too-long, broad-except
"""Optional pcgl2cache acceleration for `find_path` coordinate paths.

`find_path` with `precision_mode=false` has always returned the geometric center of each
level 2 node's chunk (`pathing.compute_rough_coordinate_path`) -- a point that is usually
nowhere near the segment itself. pcgl2cache stores a `rep_coord_nm` per level 2 node: a
representative point that does lie on the segment, served for a whole path in one request.
This module reads that when it is available and keeps the chunk center wherever it is not.

Nothing here may raise into a request. Every failure mode -- feature disabled, caveclient
not installed, service down, table without a cache, node not yet computed -- degrades to
the exact chunk-center path that was returned before, and the caller cannot tell the
difference apart from the coordinate values.

Enable per deployment by setting L2CACHE_URL (see `config.BaseConfig`); unset means the
service is never contacted.
"""

import functools
import threading
import time

import numpy as np
from flask import current_app, g

# pcgl2cache leaves an l2 id out of `rep_coord_nm` both when it has never been computed
# and when it was computed and found empty. Neither is an error and both fall back, so
# the two cases are not distinguished here.
_MAPPING_LOCK = threading.Lock()
# {server_address: (expiry_monotonic, frozenset_of_table_ids)}
_MAPPING_CACHE = {}


def _server_address():
    """Base URL of the pcgl2cache deployment, or None when the feature is off."""
    return current_app.config.get("L2CACHE_URL", None)


def _auth_token():
    """Token to present to pcgl2cache.

    Prefers the caller's token: they have already cleared `view` on this table in
    pychunkedgraph, and the l2cache `attributes` endpoint requires the same permission, so
    forwarding it avoids granting the pychunkedgraph service account `view` on every
    table. middle_auth_client also sets `g.auth_token` to None or to the literal
    "AUTH_DISABLED"; in both cases fall back to the service token.
    """
    token = getattr(g, "auth_token", None)
    if token and token != "AUTH_DISABLED":
        return token
    return current_app.config.get("AUTH_TOKEN", None)


def _client(table_id):
    """Build an L2CacheClient for `table_id`, or None if caveclient is unavailable.

    caveclient is imported here rather than at module scope so that an image built
    without it still boots and simply reports no cache.
    """
    try:
        from caveclient.auth import AuthClient
        from caveclient.l2cache import L2CacheClient
    except Exception as e:
        current_app.logger.warning(f"l2cache: caveclient unavailable ({e}); using chunk centers")
        return None

    client = L2CacheClient(
        server_address=_server_address(),
        table_name=table_id,
        auth_client=AuthClient(token=_auth_token()),
        max_retries=current_app.config.get("L2CACHE_MAX_RETRIES", 1),
    )
    # caveclient sets no timeout on any of its requests, so a black-holed l2cache would
    # hang the worker indefinitely. It never passes `timeout` itself, so binding it here
    # is safe.
    timeout = (
        current_app.config.get("L2CACHE_CONNECT_TIMEOUT_S", 1),
        current_app.config.get("L2CACHE_READ_TIMEOUT_S", 5),
    )
    client.session.request = functools.partial(client.session.request, timeout=timeout)
    return client


def _table_mapping(server_address, client):
    """Table ids pcgl2cache serves, memoized with a TTL. Empty frozenset on failure.

    `L2CacheClient.table_mapping()` memoizes on the instance and a client is built per
    request, so the TTL has to live here for the mapping to be fetched at most once per
    L2CACHE_MAPPING_TTL_S rather than once per find_path call.

    Failures are cached too, on a shorter TTL: without that, an l2cache that is down would
    make every single find_path call wait out the full request timeout before falling back.
    """
    now = time.monotonic()
    with _MAPPING_LOCK:
        entry = _MAPPING_CACHE.get(server_address)
        if entry is not None and entry[0] > now:
            return entry[1]

    try:
        tables = frozenset(client.table_mapping())
        ttl = current_app.config.get("L2CACHE_MAPPING_TTL_S", 600)
    except Exception as e:
        current_app.logger.warning(
            f"l2cache: could not read table_mapping from {server_address} ({e}); "
            "using chunk centers"
        )
        tables = frozenset()
        ttl = current_app.config.get("L2CACHE_MAPPING_FAILURE_TTL_S", 60)

    with _MAPPING_LOCK:
        _MAPPING_CACHE[server_address] = (now + ttl, tables)
    return tables


def _forget_table_mapping(server_address):
    """Drop the memoized mapping so the next request re-probes the service.

    Called when a data request fails: the service was reachable when the mapping was
    fetched but is not now, and without this every call would keep paying the timeout
    until the mapping TTL expired.
    """
    with _MAPPING_LOCK:
        _MAPPING_CACHE.pop(server_address, None)


def has_cache(table_id, client):
    """Whether pcgl2cache serves `table_id`. False on any failure."""
    return table_id in _table_mapping(_server_address(), client)


def get_rep_coords_nm(l2_ids, client):
    """Map l2 id -> representative coordinate in nm, for the ids the cache has.

    Ids the cache does not have are absent from the result. Returns {} on any failure.
    `get_l2data` is used rather than `get_l2data_table` because the latter fills missing
    coordinates with `np.empty(3)`, i.e. uninitialized values rather than NaN.
    """
    try:
        data = client.get_l2data(list(l2_ids), attributes=["rep_coord_nm"])
    except Exception as e:
        current_app.logger.warning(f"l2cache: attributes request failed ({e}); using chunk centers")
        _forget_table_mapping(_server_address())
        return {}

    coords = {}
    for l2_id, attrs in data.items():
        # keys arrive as JSON strings
        rep_coord = (attrs or {}).get("rep_coord_nm", None)
        if rep_coord is not None:
            coords[int(l2_id)] = np.array(rep_coord)
    return coords


def compute_coordinate_path(cg, l2_path):
    """Coordinate path (nm) through `l2_path`, one point per level 2 id.

    The chunk-center path is computed first as the baseline -- it is pure arithmetic with
    no I/O, and it guarantees a complete result -- then overwritten with pcgl2cache's
    representative coordinate for every id the cache can supply. So the response shape is
    identical whether or not a cache exists.
    """
    # imported here because `pathing` pulls in graph_tool, a conda-only dependency
    from ..graph.analysis import pathing

    coordinate_path = pathing.compute_rough_coordinate_path(cg, l2_path)

    if not _server_address():
        return coordinate_path

    client = _client(cg.graph_id)
    if client is None or not has_cache(cg.graph_id, client):
        return coordinate_path

    coords = get_rep_coords_nm(l2_path, client)
    if not coords:
        return coordinate_path

    for i, l2_id in enumerate(l2_path):
        rep_coord = coords.get(int(l2_id), None)
        if rep_coord is not None:
            coordinate_path[i] = rep_coord
    current_app.logger.debug(
        f"l2cache: {len(coords)}/{len(coordinate_path)} coordinates from cache for {cg.graph_id}"
    )
    return coordinate_path
