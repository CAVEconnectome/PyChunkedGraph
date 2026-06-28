"""Tests for pychunkedgraph.graph.sv_lookup.main.resolve_supervoxels_at_coords.

Uses a real bigtable-backed ChunkedGraph (gen_graph) so the hierarchy
(get_chunk_layers, get_roots, get_atomic_ids_from_coords) is real.
The seg read goes through get_local_segmentation -> meta.ws_ts_scale; we
attach a sliceable handle via graph.meta.ws_ts_scale to back
lookup_svs_from_seg with known SV ids at known coords.
"""

from math import inf
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pychunkedgraph.graph import exceptions as cg_exceptions
from pychunkedgraph.graph.sv_lookup import resolve_supervoxels_at_coords
from pychunkedgraph.graph.sv_lookup import main as sv_lookup_main

from ..helpers import create_chunk, to_label, fake_timestamp
from ...ingest.create.parent_layer import add_parent_chunk

UTC = timezone.utc


class _Read:
    def __init__(self, arr):
        self._arr = arr

    def read(self):
        return self

    def result(self):
        return self._arr


class _SliceableWS:
    """ws_ts handle stand-in: ws_ts_scale(mip)[bbox].read().result() -> (X,Y,Z,1).

    The backing seg is an in-memory uint64 array; `__getitem__` accepts a
    3-tuple of slices (the get_local_segmentation convention) and returns the
    slab with an added singleton channel axis, wrapped in a read result.
    """

    def __init__(self, shape):
        self.seg = np.zeros(shape, dtype=np.uint64)

    def __getitem__(self, key):
        slab = self.seg[key[0], key[1], key[2]][..., np.newaxis]
        return _Read(slab)

    def set_voxel(self, x, y, z, sv_id):
        self.seg[x, y, z] = np.uint64(sv_id)


def _build_two_sv_graph(gen_graph):
    """Two SVs in chunk (0,0,0) merged into one root; one SV in chunk (1,0,0)
    sharing a cross-chunk inf edge with the first → all three under one root.

    Returns (graph, sv0, sv1, sv2, root) where sv0 and sv1 are in chunk (0,0,0)
    and sv2 is in chunk (1,0,0). graph.meta._ws_cv is a _SliceableCV seeded
    with these SVs at coordinates (0,0,0), (1,0,0), (2,0,0) respectively.
    """
    graph = gen_graph(n_layers=4)
    fake_ts = fake_timestamp()

    sv0 = to_label(graph, 1, 0, 0, 0, 0)
    sv1 = to_label(graph, 1, 0, 0, 0, 1)
    sv2 = to_label(graph, 1, 1, 0, 0, 0)

    create_chunk(
        graph,
        vertices=[sv0, sv1],
        edges=[(sv0, sv1, 0.5), (sv0, sv2, inf)],
        timestamp=fake_ts,
    )
    create_chunk(
        graph,
        vertices=[sv2],
        edges=[(sv2, sv0, inf)],
        timestamp=fake_ts,
    )
    add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
    add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

    root = graph.get_root(sv0)
    assert graph.get_root(sv1) == root
    assert graph.get_root(sv2) == root

    cv = _SliceableWS(shape=(8, 8, 8))
    cv.set_voxel(0, 0, 0, sv0)
    cv.set_voxel(1, 0, 0, sv1)
    cv.set_voxel(2, 0, 0, sv2)
    graph.meta.ws_ts_scale = lambda mip=0: cv

    return graph, sv0, sv1, sv2, root


def _build_two_root_graph(gen_graph):
    """Two independent roots.

    Component A: sv0 in chunk (0,0,0) + sv2 in chunk (1,0,0) via inf edge.
    Component B: sv1 in chunk (0,0,0) — isolated (no inf cross-chunk edge).
    sv0 and sv1 share chunk (0,0,0) so are L2-siblings but not L3-merged.

    Returns (graph, sv0, sv1, sv2, root_a, root_b) with sv0, sv2 under root_a
    and sv1 under root_b. graph.meta._ws_cv is seeded with the three SVs.
    """
    graph = gen_graph(n_layers=4)
    fake_ts = fake_timestamp()

    sv0 = to_label(graph, 1, 0, 0, 0, 0)
    sv1 = to_label(graph, 1, 0, 0, 0, 1)
    sv2 = to_label(graph, 1, 1, 0, 0, 0)

    create_chunk(
        graph,
        vertices=[sv0, sv1],
        edges=[(sv0, sv2, inf)],
        timestamp=fake_ts,
    )
    create_chunk(
        graph,
        vertices=[sv2],
        edges=[(sv2, sv0, inf)],
        timestamp=fake_ts,
    )
    add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
    add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

    root_a = graph.get_root(sv0)
    root_b = graph.get_root(sv1)
    assert root_a != root_b
    assert graph.get_root(sv2) == root_a

    cv = _SliceableWS(shape=(8, 8, 8))
    cv.set_voxel(0, 0, 0, sv0)
    cv.set_voxel(1, 0, 0, sv1)
    cv.set_voxel(2, 0, 0, sv2)
    graph.meta.ws_ts_scale = lambda mip=0: cv

    return graph, sv0, sv1, sv2, root_a, root_b


class TestResolveSupervoxelsAtCoords:
    def test_2d_returns_literal_seg_sv(self, gen_graph):
        """Layer-1 node_id → orchestrator returns the literal seg SV unconditionally."""
        graph, sv0, sv1, _, _ = _build_two_sv_graph(gen_graph)
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        node_ids = np.array([sv0, sv1], dtype=np.uint64)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        np.testing.assert_array_equal(result, [sv0, sv1])

    def test_2d_on_background_returns_zero(self, gen_graph):
        """A 2D click landing on background (seg=0) returns 0 — no search."""
        graph, sv0, _, _, _ = _build_two_sv_graph(gen_graph)
        coords = np.array([[5, 5, 5]])
        node_ids = np.array([sv0], dtype=np.uint64)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert result[0] == 0

    def test_3d_interior_skips_search(self, gen_graph, monkeypatch):
        """3D coord whose literal SV's root matches the client root: literal is
        returned and cg.get_atomic_ids_from_coords is NEVER invoked."""
        graph, sv0, _, _, root = _build_two_sv_graph(gen_graph)
        coords = np.array([[0, 0, 0]])
        node_ids = np.array([root], dtype=np.uint64)

        search_calls = []
        original = graph.get_atomic_ids_from_coords

        def watcher(*a, **kw):
            search_calls.append((a, kw))
            return original(*a, **kw)

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", watcher)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert result[0] == sv0
        assert search_calls == []

    def test_3d_wrong_literal_runs_search_with_client_root(
        self, gen_graph, monkeypatch
    ):
        """3D coord whose literal seg SV resolves to a different root: orchestrator
        runs get_atomic_ids_from_coords with the *client-supplied* root and uses
        the search result."""
        graph, sv0, sv1, _, root_a, root_b = _build_two_root_graph(gen_graph)

        # Coord (0,0,0) literally holds sv0 (root_a); we claim root_b.
        coords = np.array([[0, 0, 0]])
        node_ids = np.array([root_b], dtype=np.uint64)

        # Make the search deterministic and verify it ran with root_b.
        search_calls = []

        def fake_search(coords_arg, parent_id, max_dist_nm):
            search_calls.append((np.asarray(coords_arg).tolist(), int(parent_id)))
            return np.array([sv1], dtype=np.uint64)

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", fake_search)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert result[0] == sv1
        assert len(search_calls) >= 1
        # Every call must use the client-supplied root, never the literal's root.
        assert all(parent == int(root_b) for _, parent in search_calls)

    def test_3d_literal_background_falls_into_search(self, gen_graph, monkeypatch):
        """3D coord landing on background: literal is 0, current root is 0, ≠ client
        root → orchestrator runs the parent-constrained search."""
        graph, sv0, _, _, root = _build_two_sv_graph(gen_graph)
        coords = np.array([[5, 5, 5]])  # background
        node_ids = np.array([root], dtype=np.uint64)

        captured = []

        def fake_search(coords_arg, parent_id, max_dist_nm):
            captured.append((np.asarray(coords_arg).tolist(), int(parent_id)))
            return np.array([sv0], dtype=np.uint64)

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", fake_search)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert result[0] == sv0
        assert captured, "search must run for a 3D click on background"

    def test_3d_search_exhausts_raises(self, gen_graph, monkeypatch):
        """Search returns None at every radius → BadRequest, message mentions
        the offending coord."""
        graph, _, _, _, _, root_b = _build_two_root_graph(gen_graph)
        coords = np.array([[0, 0, 0]])
        node_ids = np.array([root_b], dtype=np.uint64)

        monkeypatch.setattr(
            graph,
            "get_atomic_ids_from_coords",
            lambda *a, **kw: None,
        )

        with pytest.raises(cg_exceptions.BadRequest) as exc:
            resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert "[0, 0, 0]" in str(exc.value)

    def test_growing_radius_breaks_on_first_success(self, gen_graph, monkeypatch):
        """The growing-radius loop must stop the first time
        get_atomic_ids_from_coords returns a result — no further calls."""
        graph, sv0, _, _, _, root_b = _build_two_root_graph(gen_graph)
        coords = np.array([[0, 0, 0]])
        node_ids = np.array([root_b], dtype=np.uint64)

        calls = [0]

        def fake_search(coords_arg, parent_id, max_dist_nm):
            calls[0] += 1
            if calls[0] == 1:
                return None
            return np.array([sv0], dtype=np.uint64)

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", fake_search)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        assert result[0] == sv0
        # Exactly 2 calls: the first returned None, the second succeeded.
        # If the orchestrator kept calling after success, calls[0] would be > 2.
        assert calls[0] == 2

    def test_mixed_batch_per_root_grouping_and_positioning(
        self, gen_graph, monkeypatch
    ):
        """A single batch mixing 2D, 3D-interior, and 3D-needs-search across two
        distinct roots: exactly one search call per failing root, each receiving
        only its own coord subset; results placed at the right output indices."""
        graph, sv0, sv1, sv2, root_a, root_b = _build_two_root_graph(gen_graph)

        # idx 0: 2D click at sv0 (literal returned, root never consulted)
        # idx 1: 3D-interior; root_a; literal sv0 matches → literal returned
        # idx 2: 3D-needs-search; client says root_b but literal is sv0 (root_a)
        # idx 3: 3D-needs-search; client says root_a but literal is sv1 (root_b)
        # idx 4: 2D click at sv2 (literal returned)
        coords = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
            ]
        )
        node_ids = np.array([sv1, root_a, root_b, root_a, sv2], dtype=np.uint64)

        # The search replacements: searching for root_b finds sv1; for root_a
        # finds sv0.
        replacements = {
            int(root_b): np.array([sv1], dtype=np.uint64),
            int(root_a): np.array([sv0], dtype=np.uint64),
        }
        calls_by_root = {}

        def fake_search(coords_arg, parent_id, max_dist_nm):
            key = int(parent_id)
            calls_by_root.setdefault(key, []).append(np.asarray(coords_arg).tolist())
            return replacements[key]

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", fake_search)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)

        # 2D and 3D-interior coords pass through literally; search-resolved ones
        # come from the replacements.
        np.testing.assert_array_equal(result, [sv1, sv0, sv1, sv0, sv2])

        # Exactly two failing roots → one set of growing-radius calls per root.
        assert set(calls_by_root.keys()) == {int(root_a), int(root_b)}
        # Each root's calls received exactly its own one coord (no cross-mixing).
        for root_int, call_list in calls_by_root.items():
            for coords_in_call in call_list:
                if root_int == int(root_b):
                    assert coords_in_call == [[0, 0, 0]]
                else:
                    assert coords_in_call == [[1, 0, 0]]

    def test_all_3d_interior_skips_search_entirely(self, gen_graph, monkeypatch):
        """Cost ceiling: all 3D-interior → no search call at all."""
        graph, sv0, sv1, sv2, root = _build_two_sv_graph(gen_graph)
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        node_ids = np.array([root, root, root], dtype=np.uint64)

        called = [False]

        def watcher(*a, **kw):
            called[0] = True
            return None

        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", watcher)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        np.testing.assert_array_equal(result, [sv0, sv1, sv2])
        assert called[0] is False

    def test_all_2d_skips_root_resolution(self, gen_graph, monkeypatch):
        """Cost ceiling: all 2D → neither get_roots nor get_atomic_ids_from_coords
        is invoked. The 2D path never consults the hierarchy."""
        graph, sv0, sv1, sv2, _ = _build_two_sv_graph(gen_graph)
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        node_ids = np.array([sv0, sv1, sv2], dtype=np.uint64)

        roots_called = [False]
        search_called = [False]

        def watch_roots(*a, **kw):
            roots_called[0] = True
            raise AssertionError("get_roots must not be called for an all-2D batch")

        def watch_search(*a, **kw):
            search_called[0] = True
            raise AssertionError(
                "get_atomic_ids_from_coords must not be called for an all-2D batch"
            )

        monkeypatch.setattr(graph, "get_roots", watch_roots)
        monkeypatch.setattr(graph, "get_atomic_ids_from_coords", watch_search)

        result = resolve_supervoxels_at_coords(graph, coords, node_ids)
        np.testing.assert_array_equal(result, [sv0, sv1, sv2])
        assert roots_called[0] is False
        assert search_called[0] is False

    def test_invalid_shape_raises_before_seg_read(self, gen_graph, monkeypatch):
        """Validation fires before any seg read or hierarchy call: a 1D
        coordinates array (or wrong column count) raises BadRequest immediately."""
        graph, sv0, _, _, _ = _build_two_sv_graph(gen_graph)
        node_ids = np.array([sv0], dtype=np.uint64)

        # Trip the seg path if it ever runs, to prove we bail at validation.
        def must_not_run(*a, **kw):
            raise AssertionError("validation must fire before lookup_svs_from_seg")

        monkeypatch.setattr(sv_lookup_main, "lookup_svs_from_seg", must_not_run)

        with pytest.raises(cg_exceptions.BadRequest):
            resolve_supervoxels_at_coords(graph, np.array([0, 0, 0]), node_ids)
        with pytest.raises(cg_exceptions.BadRequest):
            resolve_supervoxels_at_coords(graph, np.array([[0, 0], [1, 1]]), node_ids)
