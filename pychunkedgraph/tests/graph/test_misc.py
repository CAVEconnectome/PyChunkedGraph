"""Tests for pychunkedgraph.graph.misc"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from pychunkedgraph.graph.misc import (
    get_latest_roots,
    get_delta_roots,
    get_proofread_root_ids,
    get_agglomerations,
    get_activated_edges,
)
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.types import Agglomeration

from ..helpers import SV, build_graph, fake_timestamp


class TestGetLatestRoots:
    def test_basic(self, gen_graph):
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
            edges=[("a0", "a1", 0.5)],
        )

        roots = get_latest_roots(cg)
        assert len(roots) >= 1

    def test_with_timestamp(self, gen_graph):
        fake_ts = fake_timestamp()
        cg, sv = build_graph(
            gen_graph,
            n_layers=4,
            supervoxels={"a0": SV()},
            timestamp=fake_ts,
        )

        roots_before = get_latest_roots(cg, fake_ts - timedelta(days=1))
        roots_after = get_latest_roots(cg)
        # Before creation, there should be no roots
        assert len(roots_before) == 0
        assert len(roots_after) >= 1


class TestGetDeltaRoots:
    def test_basic(self, gen_graph):
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=np.array([1, 1, 1]),
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )

        before_merge = datetime.now(UTC)

        cg.add_edges(
            "TestUser",
            [sv["a0"], sv["a1"]],
            affinities=[0.3],
        )

        new_roots, expired_roots = get_delta_roots(cg, before_merge)
        assert len(new_roots) >= 1


class TestGetProofreadRootIds:
    def test_after_merge(self, gen_graph):
        """After a merge, get_proofread_root_ids should return old and new root IDs."""
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=np.array([1, 1, 1]),
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )

        before_merge = datetime.now(UTC)

        # Both SVs should have separate roots before merge
        old_root0 = cg.get_root(sv["a0"])
        old_root1 = cg.get_root(sv["a1"])
        assert old_root0 != old_root1

        # Perform a merge
        cg.add_edges(
            "TestUser",
            [sv["a0"], sv["a1"]],
            affinities=[0.3],
        )

        # After merge, the two SVs share a new root
        new_root = cg.get_root(sv["a0"])
        assert new_root == cg.get_root(sv["a1"])

        old_roots, new_roots = get_proofread_root_ids(cg, start_time=before_merge)

        # The new root from the merge should appear in new_roots
        assert new_root in new_roots
        # The old roots that were merged should appear in old_roots
        old_roots_set = set(old_roots.tolist())
        assert old_root0 in old_roots_set or old_root1 in old_roots_set

    def test_empty_when_no_operations(self, gen_graph):
        """When no operations occurred, both arrays should be empty."""
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=np.array([1, 1, 1]),
            supervoxels={"a0": SV()},
        )

        # Query a time range in the future where no operations exist
        future = datetime.now(UTC) + timedelta(days=1)
        old_roots, new_roots = get_proofread_root_ids(cg, start_time=future)

        assert len(old_roots) == 0
        assert len(new_roots) == 0


class TestGetAgglomerations:
    def test_single_l2id(self):
        """Test get_agglomerations with a single L2 ID and its supervoxels."""
        l2id = np.uint64(100)
        sv1 = np.uint64(1)
        sv2 = np.uint64(2)
        sv3 = np.uint64(3)

        l2id_children_d = {l2id: np.array([sv1, sv2, sv3], dtype=np.uint64)}

        # sv_parent_d maps supervoxel -> parent l2id
        sv_parent_d = {sv1: l2id, sv2: l2id, sv3: l2id}

        # in_edges: edges within the agglomeration (sv1-sv2, sv2-sv3)
        in_edges = Edges(
            np.array([sv1, sv2], dtype=np.uint64),
            np.array([sv2, sv3], dtype=np.uint64),
        )

        # ot_edges: edges to other agglomerations (empty here)
        ot_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))

        # cx_edges: cross-chunk edges (empty here)
        cx_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))

        result = get_agglomerations(
            l2id_children_d, in_edges, ot_edges, cx_edges, sv_parent_d
        )

        assert l2id in result
        agg = result[l2id]
        assert isinstance(agg, Agglomeration)
        assert agg.node_id == l2id
        np.testing.assert_array_equal(
            agg.supervoxels, np.array([sv1, sv2, sv3], dtype=np.uint64)
        )
        # The in_edges should contain both edges (sv1-sv2, sv2-sv3) since both have node_ids1 mapping to l2id
        assert len(agg.in_edges) == 2
        assert len(agg.out_edges) == 0
        assert len(agg.cross_edges) == 0

    def test_multiple_l2ids(self):
        """Test get_agglomerations partitions edges correctly across multiple L2 IDs."""
        l2id_a = np.uint64(100)
        l2id_b = np.uint64(200)

        sv_a1 = np.uint64(1)
        sv_a2 = np.uint64(2)
        sv_b1 = np.uint64(3)
        sv_b2 = np.uint64(4)

        l2id_children_d = {
            l2id_a: np.array([sv_a1, sv_a2], dtype=np.uint64),
            l2id_b: np.array([sv_b1, sv_b2], dtype=np.uint64),
        }

        sv_parent_d = {sv_a1: l2id_a, sv_a2: l2id_a, sv_b1: l2id_b, sv_b2: l2id_b}

        # in_edges: internal edges for each agglomeration
        in_edges = Edges(
            np.array([sv_a1, sv_b1], dtype=np.uint64),
            np.array([sv_a2, sv_b2], dtype=np.uint64),
        )

        # ot_edges: edge from sv_a2 to sv_b1 (between agglomerations)
        ot_edges = Edges(
            np.array([sv_a2, sv_b1], dtype=np.uint64),
            np.array([sv_b1, sv_a2], dtype=np.uint64),
        )

        # cx_edges: empty
        cx_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))

        result = get_agglomerations(
            l2id_children_d, in_edges, ot_edges, cx_edges, sv_parent_d
        )

        assert len(result) == 2
        assert l2id_a in result
        assert l2id_b in result

        agg_a = result[l2id_a]
        agg_b = result[l2id_b]

        # Each agglomeration should have exactly 1 internal edge
        assert len(agg_a.in_edges) == 1
        assert len(agg_b.in_edges) == 1

        # Each agglomeration should have exactly 1 out_edge
        assert len(agg_a.out_edges) == 1
        assert len(agg_b.out_edges) == 1

    def test_empty_edges(self):
        """Test get_agglomerations with an L2 ID that has no edges at all."""
        l2id = np.uint64(50)
        sv = np.uint64(10)

        l2id_children_d = {l2id: np.array([sv], dtype=np.uint64)}
        sv_parent_d = {sv: l2id}

        in_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))
        ot_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))
        cx_edges = Edges(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))

        result = get_agglomerations(
            l2id_children_d, in_edges, ot_edges, cx_edges, sv_parent_d
        )

        assert l2id in result
        agg = result[l2id]
        assert agg.node_id == l2id
        assert len(agg.in_edges) == 0
        assert len(agg.out_edges) == 0
        assert len(agg.cross_edges) == 0


class TestGetActivatedEdges:
    @pytest.mark.timeout(30)
    def test_returns_numpy_array_after_merge(self, gen_graph):
        """After merging two isolated SVs, get_activated_edges returns a numpy array."""
        cg, sv = build_graph(
            gen_graph,
            n_layers=2,
            atomic_chunk_bounds=np.array([1, 1, 1]),
            supervoxels={"a0": SV(), "a1": SV(seg=1)},
        )

        # Merge the two isolated supervoxels
        result = cg.add_edges(
            "TestUser",
            [sv["a0"], sv["a1"]],
            affinities=[0.3],
        )

        activated = get_activated_edges(cg, result.operation_id)
        assert isinstance(activated, np.ndarray)
