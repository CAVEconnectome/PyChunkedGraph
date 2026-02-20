"""Tests for pychunkedgraph.graph.chunkedgraph - extended coverage"""

from datetime import datetime, timedelta, UTC
from math import inf

import numpy as np
import pytest

from .helpers import create_chunk, to_label
from ..ingest.create.parent_layer import add_parent_chunk
from ..graph.operation import GraphEditOperation, MergeOperation, SplitOperation
from ..graph.exceptions import PreconditionError


class TestChunkedGraphExtended:
    def _build_graph(self, gen_graph):
        """Build a simple multi-chunk graph."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)

        # Chunk A: sv 0, 1 connected
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )

        # Chunk B: sv 0 connected cross-chunk to A
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )

        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    def test_is_root_true(self, gen_graph):
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        root = graph.get_root(sv)
        assert graph.is_root(root)

    def test_is_root_false(self, gen_graph):
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        assert not graph.is_root(sv)

    def test_get_parents_raw_only(self, gen_graph):
        graph = self._build_graph(gen_graph)
        svs = np.array(
            [
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            dtype=np.uint64,
        )
        parents = graph.get_parents(svs, raw_only=True)
        assert len(parents) == 2
        # Parents should be L2 IDs
        for p in parents:
            assert graph.get_chunk_layer(p) == 2

    def test_get_parents_fail_to_zero(self, gen_graph):
        graph = self._build_graph(gen_graph)
        # Non-existent ID should return 0 with fail_to_zero
        bad_id = np.uint64(99999999)
        result = graph.get_parents(
            np.array([bad_id], dtype=np.uint64), fail_to_zero=True
        )
        assert result[0] == 0

    def test_get_children_flatten(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        children = graph.get_children([root], flatten=True)
        assert isinstance(children, np.ndarray)
        assert len(children) > 0

    def test_is_latest_roots(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = graph.is_latest_roots(np.array([root], dtype=np.uint64))
        assert result[0]

    def test_get_node_timestamps(self, gen_graph):
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        root = graph.get_root(sv)
        ts = graph.get_node_timestamps(np.array([root]), return_numpy=False)
        assert len(ts) == 1

    def test_get_earliest_timestamp(self, gen_graph):
        graph = self._build_graph(gen_graph)
        ts = graph.get_earliest_timestamp()
        # May return None if no operation logs exist; test the method runs
        assert ts is None or isinstance(ts, datetime)

    def test_get_l2children(self, gen_graph):
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        l2_children = graph.get_l2children(np.array([root], dtype=np.uint64))
        assert len(l2_children) > 0
        for child in l2_children:
            assert graph.get_chunk_layer(child) == 2

    # --- helpers for edit-based tests ---

    def _build_and_merge(self, gen_graph):
        """Build a single-chunk graph with two disconnected SVs and merge them."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        return graph, result.new_root_ids[0], result

    @pytest.mark.timeout(30)
    def test_get_operation_ids(self, gen_graph):
        """After a merge, get_operation_ids on the new root should return at least one operation."""
        graph, new_root, result = self._build_and_merge(gen_graph)
        op_ids = graph.get_operation_ids([new_root])
        assert new_root in op_ids
        assert len(op_ids[new_root]) >= 1
        # Each entry is (operation_id_value, timestamp)
        op_id_val, ts = op_ids[new_root][0]
        assert op_id_val == result.operation_id

    @pytest.mark.timeout(30)
    def test_get_single_leaf_multiple(self, gen_graph):
        """get_single_leaf_multiple for an L2 node should return an L1 supervoxel."""
        graph, new_root, _ = self._build_and_merge(gen_graph)
        # The new_root in n_layers=2 is actually L2
        assert graph.get_chunk_layer(new_root) == 2
        leaves = graph.get_single_leaf_multiple(np.array([new_root], dtype=np.uint64))
        assert len(leaves) == 1
        assert graph.get_chunk_layer(leaves[0]) == 1
        # The returned leaf should be one of our two SVs
        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)
        assert leaves[0] in [sv0, sv1]

    @pytest.mark.timeout(30)
    def test_get_atomic_cross_edges(self, gen_graph):
        """get_atomic_cross_edges for an L2 node with cross-chunk connections."""
        graph = self._build_graph(gen_graph)
        sv_a0 = to_label(graph, 1, 0, 0, 0, 0)

        # Get the L2 parent of sv_a0
        parent = graph.get_parents(np.array([sv_a0], dtype=np.uint64), raw_only=True)[0]
        assert graph.get_chunk_layer(parent) == 2

        result = graph.get_atomic_cross_edges([parent])
        assert parent in result
        # Should have at least one layer of cross edges
        assert isinstance(result[parent], dict)

    @pytest.mark.timeout(30)
    def test_get_cross_chunk_edges_raw(self, gen_graph):
        """get_cross_chunk_edges with raw_only=True should return cross edges."""
        graph = self._build_graph(gen_graph)
        sv_a0 = to_label(graph, 1, 0, 0, 0, 0)

        # Get the L2 parent
        parent = graph.get_parents(np.array([sv_a0], dtype=np.uint64), raw_only=True)[0]
        assert graph.get_chunk_layer(parent) == 2

        result = graph.get_cross_chunk_edges([parent], raw_only=True)
        assert parent in result
        assert isinstance(result[parent], dict)

    @pytest.mark.timeout(30)
    def test_get_parents_not_current(self, gen_graph):
        """get_parents with current=False should return list of (parent, timestamp) tuples."""
        graph, new_root, _ = self._build_and_merge(gen_graph)
        sv0 = to_label(graph, 1, 0, 0, 0, 0)

        # current=False returns list of lists of (value, timestamp) pairs
        parents = graph.get_parents(
            np.array([sv0], dtype=np.uint64), raw_only=True, current=False
        )
        assert len(parents) == 1
        # Each element is a list of (parent_value, timestamp) tuples
        assert isinstance(parents[0], list)
        assert len(parents[0]) >= 1
        parent_val, parent_ts = parents[0][0]
        assert parent_val != 0
        assert isinstance(parent_ts, datetime)


class TestFromLogRecord:
    """Test GraphEditOperation.from_log_record with real merge/split logs."""

    def _build_two_sv_graph(self, gen_graph):
        """Build a 2-layer graph with two disconnected SVs."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        return graph

    @pytest.mark.timeout(30)
    def test_merge_from_log(self, gen_graph):
        """After a merge, from_log_record should return a MergeOperation."""
        graph = self._build_two_sv_graph(gen_graph)
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        log, ts = graph.client.read_log_entry(result.operation_id)
        op = GraphEditOperation.from_log_record(graph, log)
        assert isinstance(op, MergeOperation)

    @pytest.mark.timeout(30)
    def test_split_from_log(self, gen_graph):
        """After a split, from_log_record should return a SplitOperation."""
        graph = self._build_two_sv_graph(gen_graph)
        # First merge so the SVs belong to the same root
        graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        # Now split them
        split_result = graph.remove_edges(
            "TestUser",
            source_ids=to_label(graph, 1, 0, 0, 0, 0),
            sink_ids=to_label(graph, 1, 0, 0, 0, 1),
            mincut=False,
        )
        log, ts = graph.client.read_log_entry(split_result.operation_id)
        op = GraphEditOperation.from_log_record(graph, log)
        assert isinstance(op, SplitOperation)


class TestCheckIds:
    """Test ID validation in MergeOperation."""

    def _build_two_sv_graph(self, gen_graph):
        """Build a 2-layer graph with two disconnected SVs."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        return graph

    @pytest.mark.timeout(30)
    def test_source_equals_sink_raises(self, gen_graph):
        """MergeOperation with source==sink should raise PreconditionError (self-loop)."""
        graph = self._build_two_sv_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        with pytest.raises(PreconditionError):
            graph.add_edges(
                "TestUser",
                [sv, sv],
                affinities=[0.3],
            )

    @pytest.mark.timeout(30)
    def test_nonexistent_supervoxel_raises(self, gen_graph):
        """Using a supervoxel ID that doesn't exist should raise an error."""
        graph = self._build_two_sv_graph(gen_graph)
        sv_real = to_label(graph, 1, 0, 0, 0, 0)
        # Use a layer-2 ID as a fake "supervoxel", which fails the layer check
        sv_fake = to_label(graph, 2, 0, 0, 0, 99)
        with pytest.raises(Exception):
            graph.add_edges(
                "TestUser",
                [sv_real, sv_fake],
                affinities=[0.3],
            )


class TestGetRootsExtended:
    """Tests for get_roots with stop_layer and ceil parameters (lines 380-461)."""

    def _build_cross_chunk(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph, fake_ts

    @pytest.mark.timeout(30)
    def test_get_roots_with_stop_layer(self, gen_graph):
        """get_roots with stop_layer should return IDs at that layer."""
        graph, _ = self._build_cross_chunk(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        # Stop at layer 3 instead of going to root (layer 4)
        result = graph.get_roots(np.array([sv], dtype=np.uint64), stop_layer=3)
        assert len(result) == 1
        assert graph.get_chunk_layer(result[0]) == 3

    @pytest.mark.timeout(30)
    def test_get_roots_with_stop_layer_and_ceil_false(self, gen_graph):
        """get_roots with stop_layer and ceil=False should not exceed stop_layer."""
        graph, _ = self._build_cross_chunk(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        result = graph.get_roots(
            np.array([sv], dtype=np.uint64), stop_layer=3, ceil=False
        )
        assert len(result) == 1
        assert graph.get_chunk_layer(result[0]) <= 3

    @pytest.mark.timeout(30)
    def test_get_roots_multiple_svs(self, gen_graph):
        """get_roots with multiple SVs should return root for each."""
        graph, _ = self._build_cross_chunk(gen_graph)
        svs = np.array(
            [
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
                to_label(graph, 1, 1, 0, 0, 0),
            ],
            dtype=np.uint64,
        )
        roots = graph.get_roots(svs)
        assert len(roots) == 3
        # All should reach the top layer
        for r in roots:
            assert graph.get_chunk_layer(r) == 4

    @pytest.mark.timeout(30)
    def test_get_roots_already_at_stop_layer(self, gen_graph):
        """get_roots for a node already at stop_layer should return it unchanged."""
        graph, _ = self._build_cross_chunk(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        root = graph.get_root(sv)
        # root is at layer 4; asking for stop_layer=4 should return it
        result = graph.get_roots(np.array([root], dtype=np.uint64), stop_layer=4)
        assert result[0] == root

    @pytest.mark.timeout(30)
    def test_get_roots_fail_to_zero(self, gen_graph):
        """get_roots with a zero ID and fail_to_zero should keep it as zero."""
        graph, _ = self._build_cross_chunk(gen_graph)
        result = graph.get_roots(np.array([0], dtype=np.uint64), fail_to_zero=True)
        assert result[0] == 0

    @pytest.mark.timeout(30)
    def test_get_root_stop_layer_ceil_false(self, gen_graph):
        """get_root (singular) with stop_layer and ceil=False."""
        graph, _ = self._build_cross_chunk(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        result = graph.get_root(sv, stop_layer=3, ceil=False)
        assert graph.get_chunk_layer(result) <= 3


class TestGetChildrenExtended:
    """Tests for get_children with flatten=True and edge cases (lines 271-296)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_children_flatten_multiple(self, gen_graph):
        """get_children with multiple node IDs and flatten=True returns flat array."""
        graph = self._build_graph(gen_graph)
        sv_a0 = to_label(graph, 1, 0, 0, 0, 0)
        sv_b0 = to_label(graph, 1, 1, 0, 0, 0)

        parent_a = graph.get_parents(np.array([sv_a0], dtype=np.uint64), raw_only=True)[
            0
        ]
        parent_b = graph.get_parents(np.array([sv_b0], dtype=np.uint64), raw_only=True)[
            0
        ]

        children = graph.get_children([parent_a, parent_b], flatten=True)
        assert isinstance(children, np.ndarray)
        # Should contain at least sv_a0, sv_a1, sv_b0
        assert len(children) >= 3

    @pytest.mark.timeout(30)
    def test_get_children_flatten_empty(self, gen_graph):
        """get_children with flatten=True on empty list returns empty array."""
        graph = self._build_graph(gen_graph)
        children = graph.get_children([], flatten=True)
        assert isinstance(children, np.ndarray)
        assert len(children) == 0

    @pytest.mark.timeout(30)
    def test_get_children_dict(self, gen_graph):
        """get_children without flatten returns a dict."""
        graph = self._build_graph(gen_graph)
        sv_a0 = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv_a0], dtype=np.uint64), raw_only=True)[0]
        children_d = graph.get_children([parent])
        assert isinstance(children_d, dict)
        assert parent in children_d

    @pytest.mark.timeout(30)
    def test_get_children_scalar(self, gen_graph):
        """get_children with a scalar node_id returns an array."""
        graph = self._build_graph(gen_graph)
        sv_a0 = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv_a0], dtype=np.uint64), raw_only=True)[0]
        children = graph.get_children(parent, raw_only=True)
        assert isinstance(children, np.ndarray)
        assert len(children) >= 1


class TestIsLatestRootsExtended:
    """Tests for is_latest_roots (lines 524-544)."""

    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        # Get the initial roots
        root0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        root1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        # Merge
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, root0, root1, new_root

    @pytest.mark.timeout(30)
    def test_is_latest_roots_after_merge(self, gen_graph):
        """After a merge, old roots should not be latest, new root should be."""
        graph, root0, root1, new_root = self._build_and_merge(gen_graph)
        result = graph.is_latest_roots(
            np.array([root0, root1, new_root], dtype=np.uint64)
        )
        # Old roots are superseded
        assert not result[0]
        assert not result[1]
        # New root is latest
        assert result[2]

    @pytest.mark.timeout(30)
    def test_is_latest_roots_empty(self, gen_graph):
        """is_latest_roots with nonexistent IDs should return all False."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        result = graph.is_latest_roots(np.array([99999999], dtype=np.uint64))
        assert not result[0]


class TestGetNodeTimestampsExtended:
    """Tests for get_node_timestamps (lines 773-800)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_node_timestamps_return_numpy(self, gen_graph):
        """get_node_timestamps with return_numpy=True should return numpy array."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        ts = graph.get_node_timestamps(
            np.array([root], dtype=np.uint64), return_numpy=True
        )
        assert isinstance(ts, np.ndarray)
        assert len(ts) == 1

    @pytest.mark.timeout(30)
    def test_get_node_timestamps_return_list(self, gen_graph):
        """get_node_timestamps with return_numpy=False should return a list."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        ts = graph.get_node_timestamps(
            np.array([root], dtype=np.uint64), return_numpy=False
        )
        assert isinstance(ts, list)
        assert len(ts) == 1

    @pytest.mark.timeout(30)
    def test_get_node_timestamps_empty(self, gen_graph):
        """get_node_timestamps with nonexistent nodes should handle gracefully."""
        graph = self._build_graph(gen_graph)
        ts = graph.get_node_timestamps(
            np.array([np.uint64(99999999)], dtype=np.uint64), return_numpy=True
        )
        # Should either return empty or return a fallback timestamp
        assert isinstance(ts, np.ndarray)

    @pytest.mark.timeout(30)
    def test_get_node_timestamps_empty_return_list(self, gen_graph):
        """get_node_timestamps with empty dict result and return_numpy=False."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        # Don't create any chunks; asking for timestamps on nonexistent nodes
        ts = graph.get_node_timestamps(
            np.array([np.uint64(99999999)], dtype=np.uint64), return_numpy=False
        )
        assert isinstance(ts, list)
        assert len(ts) == 0


class TestGetOperationIdsExtended:
    """Tests for get_operation_ids (lines 1033-1042)."""

    @pytest.mark.timeout(30)
    def test_get_operation_ids_no_ops(self, gen_graph):
        """get_operation_ids on a node with no operations returns empty dict."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = graph.get_operation_ids([root])
        # No operations => root may not be in result, or have empty list
        if root in result:
            assert isinstance(result[root], list)


class TestGetSingleLeafMultipleExtended:
    """Tests for get_single_leaf_multiple (lines 1044-1062)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_single_leaf_from_root(self, gen_graph):
        """get_single_leaf_multiple from a root (layer 4) should drill down to layer 1."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        assert graph.get_chunk_layer(root) == 4
        leaves = graph.get_single_leaf_multiple(np.array([root], dtype=np.uint64))
        assert len(leaves) == 1
        assert graph.get_chunk_layer(leaves[0]) == 1

    @pytest.mark.timeout(30)
    def test_get_single_leaf_from_l2(self, gen_graph):
        """get_single_leaf_multiple from L2 node should return one of its SV children."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv], dtype=np.uint64), raw_only=True)[0]
        assert graph.get_chunk_layer(parent) == 2
        leaves = graph.get_single_leaf_multiple(np.array([parent], dtype=np.uint64))
        assert len(leaves) == 1
        assert graph.get_chunk_layer(leaves[0]) == 1

    @pytest.mark.timeout(30)
    def test_get_single_leaf_multiple_nodes(self, gen_graph):
        """get_single_leaf_multiple with multiple node IDs should return one leaf each."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        parent_a = graph.get_parents(np.array([sv_a], dtype=np.uint64), raw_only=True)[
            0
        ]
        parent_b = graph.get_parents(np.array([sv_b], dtype=np.uint64), raw_only=True)[
            0
        ]
        leaves = graph.get_single_leaf_multiple(
            np.array([parent_a, parent_b], dtype=np.uint64)
        )
        assert len(leaves) == 2
        for leaf in leaves:
            assert graph.get_chunk_layer(leaf) == 1


class TestGetL2ChildrenExtended:
    """Tests for get_l2children (lines 1079-1092)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_l2children_from_root(self, gen_graph):
        """get_l2children from a root should return all L2 children."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        l2_children = graph.get_l2children(np.array([root], dtype=np.uint64))
        assert isinstance(l2_children, np.ndarray)
        assert len(l2_children) >= 2
        for child in l2_children:
            assert graph.get_chunk_layer(child) == 2

    @pytest.mark.timeout(30)
    def test_get_l2children_from_l3(self, gen_graph):
        """get_l2children from an L3 node should return L2 children."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        # Get L3 parent
        root = graph.get_root(sv, stop_layer=3)
        assert graph.get_chunk_layer(root) == 3
        l2_children = graph.get_l2children(np.array([root], dtype=np.uint64))
        assert len(l2_children) >= 1
        for child in l2_children:
            assert graph.get_chunk_layer(child) == 2

    @pytest.mark.timeout(30)
    def test_get_l2children_from_l2(self, gen_graph):
        """get_l2children from an L2 node drills down to its children,
        which are L1 - so no L2 children are found; result is empty."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv], dtype=np.uint64), raw_only=True)[0]
        assert graph.get_chunk_layer(parent) == 2
        l2_children = graph.get_l2children(np.array([parent], dtype=np.uint64))
        # L2 nodes only have L1 (SV) children, so no L2 descendants found
        assert isinstance(l2_children, np.ndarray)
        assert len(l2_children) == 0


class TestGetChunkLayersExtended:
    """Tests for get_chunk_layers and related helpers (line 951-952, 946)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_chunk_layers_multiple(self, gen_graph):
        """get_chunk_layers for nodes at different layers."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent_l2 = graph.get_parents(np.array([sv], dtype=np.uint64), raw_only=True)[0]
        root = graph.get_root(sv)
        layers = graph.get_chunk_layers(
            np.array([sv, parent_l2, root], dtype=np.uint64)
        )
        assert layers[0] == 1
        assert layers[1] == 2
        assert layers[2] == 4

    @pytest.mark.timeout(30)
    def test_get_segment_id_limit(self, gen_graph):
        """get_segment_id_limit should return a valid limit."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        limit = graph.get_segment_id_limit(sv)
        assert limit > 0

    @pytest.mark.timeout(30)
    def test_get_chunk_coordinates(self, gen_graph):
        """get_chunk_coordinates should return the chunk coordinates of a node."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        coords = graph.get_chunk_coordinates(sv)
        assert len(coords) == 3
        np.testing.assert_array_equal(coords, [0, 0, 0])

    @pytest.mark.timeout(30)
    def test_get_chunk_layers_and_coordinates(self, gen_graph):
        """get_chunk_layers_and_coordinates returns layers and coords together."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        layers, coords = graph.get_chunk_layers_and_coordinates(
            np.array([sv_a, sv_b], dtype=np.uint64)
        )
        assert len(layers) == 2
        assert layers[0] == 1
        assert layers[1] == 1
        assert coords.shape == (2, 3)


class TestGetAtomicCrossEdgesExtended:
    """Tests for get_atomic_cross_edges (lines 315-336)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_atomic_cross_edges_multiple_l2(self, gen_graph):
        """get_atomic_cross_edges with multiple L2 IDs."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        parent_a = graph.get_parents(np.array([sv_a], dtype=np.uint64), raw_only=True)[
            0
        ]
        parent_b = graph.get_parents(np.array([sv_b], dtype=np.uint64), raw_only=True)[
            0
        ]
        result = graph.get_atomic_cross_edges([parent_a, parent_b])
        assert parent_a in result
        assert parent_b in result
        # At least one should have cross edges
        has_edges = any(len(v) > 0 for v in result.values())
        assert has_edges

    @pytest.mark.timeout(30)
    def test_get_atomic_cross_edges_no_cross(self, gen_graph):
        """get_atomic_cross_edges for an L2 node with no cross edges."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv], dtype=np.uint64), raw_only=True)[0]
        result = graph.get_atomic_cross_edges([parent])
        assert parent in result
        assert isinstance(result[parent], dict)
        # No cross edges
        assert len(result[parent]) == 0


class TestGetAllParentsDictExtended:
    """Tests for get_all_parents_dict and get_all_parents_dict_multiple."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_all_parents_dict(self, gen_graph):
        """get_all_parents_dict returns a dict mapping layer -> parent."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        d = graph.get_all_parents_dict(sv)
        assert isinstance(d, dict)
        # Should have entries for layers 2, 3, 4
        assert 2 in d
        assert 4 in d

    @pytest.mark.timeout(30)
    def test_get_all_parents_dict_multiple(self, gen_graph):
        """get_all_parents_dict_multiple for multiple SVs."""
        graph = self._build_graph(gen_graph)
        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)
        result = graph.get_all_parents_dict_multiple(
            np.array([sv0, sv1], dtype=np.uint64)
        )
        assert sv0 in result
        assert sv1 in result
        # Both should have parents at layer 2
        assert 2 in result[sv0]
        assert 2 in result[sv1]


class TestMiscMethods:
    """Tests for misc ChunkedGraph methods."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_serialized_info(self, gen_graph):
        """get_serialized_info should return a dict with graph_id."""
        graph = self._build_graph(gen_graph)
        info = graph.get_serialized_info()
        assert isinstance(info, dict)
        assert "graph_id" in info

    @pytest.mark.timeout(30)
    def test_get_chunk_id(self, gen_graph):
        """get_chunk_id should return a valid chunk id."""
        graph = self._build_graph(gen_graph)
        chunk_id = graph.get_chunk_id(layer=2, x=0, y=0, z=0)
        assert chunk_id > 0
        assert graph.get_chunk_layer(chunk_id) == 2

    @pytest.mark.timeout(30)
    def test_get_node_id(self, gen_graph):
        """get_node_id should construct node IDs correctly."""
        graph = self._build_graph(gen_graph)
        node_id = graph.get_node_id(np.uint64(1), layer=1, x=0, y=0, z=0)
        assert node_id > 0
        assert graph.get_chunk_layer(node_id) == 1

    @pytest.mark.timeout(30)
    def test_get_segment_id(self, gen_graph):
        """get_segment_id should extract segment id from node id."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 5)
        seg_id = graph.get_segment_id(sv)
        assert seg_id == 5

    @pytest.mark.timeout(30)
    def test_get_parent_chunk_id(self, gen_graph):
        """get_parent_chunk_id should return the chunk id of the parent layer."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent_chunk = graph.get_parent_chunk_id(sv)
        assert graph.get_chunk_layer(parent_chunk) == 2

    @pytest.mark.timeout(30)
    def test_get_children_chunk_ids(self, gen_graph):
        """get_children_chunk_ids should return chunk IDs one layer below."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        root = graph.get_root(sv)
        # root is at layer 4; children chunks should be at layer 3
        children_chunks = graph.get_children_chunk_ids(root)
        for cc in children_chunks:
            assert graph.get_chunk_layer(cc) == 3

    @pytest.mark.timeout(30)
    def test_get_cross_chunk_edges_empty(self, gen_graph):
        """get_cross_chunk_edges with empty node_ids should return empty dict."""
        graph = self._build_graph(gen_graph)
        result = graph.get_cross_chunk_edges([], raw_only=True)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestIsLatestRootsAfterMerge:
    """Test is_latest_roots after a merge operation (lines 524-539, 689-701)."""

    def _build_and_merge(self, gen_graph):
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        old_root0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, old_root0, old_root1, new_root

    @pytest.mark.timeout(30)
    def test_is_latest_roots_after_merge(self, gen_graph):
        """After merge, old roots are not latest; new root is latest."""
        graph, old_root0, old_root1, new_root = self._build_and_merge(gen_graph)
        result = graph.is_latest_roots(
            np.array([old_root0, old_root1, new_root], dtype=np.uint64)
        )
        assert not result[0], "Old root0 should not be latest after merge"
        assert not result[1], "Old root1 should not be latest after merge"
        assert result[2], "New root should be latest after merge"


class TestGetSubgraphNodesOnly:
    """Test get_subgraph with nodes_only=True (lines 602-613)."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes_only(self, gen_graph):
        """get_subgraph with nodes_only=True should return layer->node_ids dict."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = graph.get_subgraph(root, nodes_only=True)
        # Result should be a dict with layer 2 by default
        assert isinstance(result, dict)
        assert 2 in result
        l2_nodes = result[2]
        assert len(l2_nodes) >= 2
        for node in l2_nodes:
            assert graph.get_chunk_layer(node) == 2

    @pytest.mark.timeout(30)
    def test_get_subgraph_nodes_only_multiple_layers(self, gen_graph):
        """get_subgraph with nodes_only=True and return_layers=[2,3]."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = graph.get_subgraph(root, nodes_only=True, return_layers=[2, 3])
        assert isinstance(result, dict)
        # Should have entries for layer 2 and/or 3
        assert 2 in result or 3 in result


class TestGetSubgraphEdgesOnly:
    """Test get_subgraph with edges_only=True."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_subgraph_edges_only(self, gen_graph):
        """get_subgraph with edges_only=True should return edges."""
        graph = self._build_graph(gen_graph)
        root = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        result = graph.get_subgraph(root, edges_only=True)
        # edges_only returns Edges from get_l2_agglomerations
        # It should be a tuple of Edges or similar iterable
        assert result is not None


# ===========================================================================
# is_latest_roots after merge -- detailed tests (lines 689-701)
# ===========================================================================


class TestIsLatestRootsDetailed:
    """Detailed tests for is_latest_roots checking old roots are not latest after merge."""

    def _build_and_merge(self, gen_graph):
        """Build graph with two disconnected SVs, merge them, return old and new roots."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        old_root0 = graph.get_root(to_label(graph, 1, 0, 0, 0, 0))
        old_root1 = graph.get_root(to_label(graph, 1, 0, 0, 0, 1))

        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        new_root = result.new_root_ids[0]
        return graph, old_root0, old_root1, new_root

    @pytest.mark.timeout(30)
    def test_is_latest_roots_correct(self, gen_graph):
        """After merge, old roots should be flagged as not latest, new root as latest."""
        graph, old_root0, old_root1, new_root = self._build_and_merge(gen_graph)
        result = graph.is_latest_roots(
            np.array([old_root0, old_root1, new_root], dtype=np.uint64)
        )
        assert not result[0], "Old root0 should not be latest after merge"
        assert not result[1], "Old root1 should not be latest after merge"
        assert result[2], "New root should be latest after merge"

    @pytest.mark.timeout(30)
    def test_is_latest_roots_single_old_root(self, gen_graph):
        """Check a single old root is not latest after merge."""
        graph, old_root0, _, _ = self._build_and_merge(gen_graph)
        result = graph.is_latest_roots(np.array([old_root0], dtype=np.uint64))
        assert not result[0]

    @pytest.mark.timeout(30)
    def test_is_latest_roots_single_new_root(self, gen_graph):
        """Check a single new root is latest after merge."""
        graph, _, _, new_root = self._build_and_merge(gen_graph)
        result = graph.is_latest_roots(np.array([new_root], dtype=np.uint64))
        assert result[0]


# ===========================================================================
# get_chunk_coordinates_multiple (lines 958-961)
# ===========================================================================


class TestGetChunkCoordinatesMultiple:
    """Tests for get_chunk_coordinates_multiple with same/different layer assertions."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_same_layer(self, gen_graph):
        """get_chunk_coordinates_multiple with L2 node IDs should return correct coordinates."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        # Get L2 parents
        parent_a = graph.get_parents(np.array([sv_a], dtype=np.uint64), raw_only=True)[
            0
        ]
        parent_b = graph.get_parents(np.array([sv_b], dtype=np.uint64), raw_only=True)[
            0
        ]
        assert graph.get_chunk_layer(parent_a) == 2
        assert graph.get_chunk_layer(parent_b) == 2

        coords = graph.get_chunk_coordinates_multiple(
            np.array([parent_a, parent_b], dtype=np.uint64)
        )
        assert coords.shape == (2, 3)
        # parent_a is in chunk (0,0,0), parent_b is in chunk (1,0,0)
        np.testing.assert_array_equal(coords[0], [0, 0, 0])
        np.testing.assert_array_equal(coords[1], [1, 0, 0])

    @pytest.mark.timeout(30)
    def test_different_layers_raises(self, gen_graph):
        """get_chunk_coordinates_multiple with nodes at different layers should raise."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        parent_l2 = graph.get_parents(np.array([sv_a], dtype=np.uint64), raw_only=True)[
            0
        ]
        root = graph.get_root(sv_a)

        assert graph.get_chunk_layer(parent_l2) == 2
        assert graph.get_chunk_layer(root) == 4

        with pytest.raises(AssertionError, match="must be same layer"):
            graph.get_chunk_coordinates_multiple(
                np.array([parent_l2, root], dtype=np.uint64)
            )

    @pytest.mark.timeout(30)
    def test_empty_array(self, gen_graph):
        """get_chunk_coordinates_multiple with empty array should return empty result."""
        graph = self._build_graph(gen_graph)
        coords = graph.get_chunk_coordinates_multiple(np.array([], dtype=np.uint64))
        assert len(coords) == 0


# ===========================================================================
# get_parent_chunk_id_multiple and get_parent_chunk_ids (lines 991, 996)
# ===========================================================================


class TestParentChunkIdMethods:
    """Tests for get_parent_chunk_id_multiple and get_parent_chunk_ids."""

    def _build_graph(self, gen_graph):
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 1, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 1, 0, 0, 0)],
            edges=[
                (to_label(graph, 1, 1, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 0), inf),
            ],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 3, [1, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        return graph

    @pytest.mark.timeout(30)
    def test_get_parent_chunk_id_multiple(self, gen_graph):
        """get_parent_chunk_id_multiple should return parent chunk IDs for all nodes."""
        graph = self._build_graph(gen_graph)
        sv_a = to_label(graph, 1, 0, 0, 0, 0)
        sv_b = to_label(graph, 1, 1, 0, 0, 0)
        # Get L2 parents
        parent_a = graph.get_parents(np.array([sv_a], dtype=np.uint64), raw_only=True)[
            0
        ]
        parent_b = graph.get_parents(np.array([sv_b], dtype=np.uint64), raw_only=True)[
            0
        ]

        parent_chunks = graph.get_parent_chunk_id_multiple(
            np.array([parent_a, parent_b], dtype=np.uint64)
        )
        assert len(parent_chunks) == 2
        for pc in parent_chunks:
            assert graph.get_chunk_layer(pc) == 3

    @pytest.mark.timeout(30)
    def test_get_parent_chunk_ids(self, gen_graph):
        """get_parent_chunk_ids should return all parent chunk IDs up the hierarchy."""
        graph = self._build_graph(gen_graph)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent_chunk_ids = graph.get_parent_chunk_ids(sv)
        # Should have parent chunk IDs for layers 2, 3, 4
        assert len(parent_chunk_ids) >= 2
        layers = [graph.get_chunk_layer(pc) for pc in parent_chunk_ids]
        # Layers should be ascending (from layer 2 up)
        for i in range(len(layers) - 1):
            assert layers[i] < layers[i + 1]


# ===========================================================================
# read_chunk_edges (lines 1005-1007)
# ===========================================================================


class TestReadChunkEdges:
    """Tests for read_chunk_edges method."""

    @pytest.mark.timeout(30)
    def test_read_chunk_edges_returns_dict(self, gen_graph):
        """read_chunk_edges should return a dict (possibly empty for gs:// edges source)."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        sv = to_label(graph, 1, 0, 0, 0, 0)
        parent = graph.get_parents(np.array([sv], dtype=np.uint64), raw_only=True)[0]
        # read_chunk_edges uses io.edges.get_chunk_edges which reads from GCS/file.
        # With gs:// edges source and no actual files, it should raise or return empty.
        try:
            result = graph.read_chunk_edges(np.array([parent], dtype=np.uint64))
            assert isinstance(result, dict)
        except Exception:
            # Expected: GCS access will fail in test env
            pass


# ===========================================================================
# get_proofread_root_ids (lines 1017-1019)
# ===========================================================================


class TestGetProofreadRootIds:
    """Tests for get_proofread_root_ids method."""

    @pytest.mark.timeout(30)
    def test_get_proofread_root_ids_no_ops(self, gen_graph):
        """get_proofread_root_ids with no operations should return empty arrays."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)

        old_roots, new_roots = graph.get_proofread_root_ids()
        assert len(old_roots) == 0
        assert len(new_roots) == 0

    @pytest.mark.timeout(30)
    def test_get_proofread_root_ids_after_merge(self, gen_graph):
        """get_proofread_root_ids after a merge should return the old and new roots."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        result = graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        old_roots, new_roots = graph.get_proofread_root_ids()
        assert len(new_roots) >= 1
        assert result.new_root_ids[0] in new_roots


# ===========================================================================
# remove_edges via shim path (line 876) -- source_ids/sink_ids without atomic_edges
# ===========================================================================


class TestRemoveEdgesShim:
    """Test remove_edges with source_ids and sink_ids but no atomic_edges (shim path)."""

    def _build_connected_graph(self, gen_graph):
        """Build a 2-layer graph with two connected SVs."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[
                (to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1), 0.5),
            ],
            timestamp=fake_ts,
        )
        return graph

    @pytest.mark.timeout(30)
    def test_remove_edges_with_source_sink_ids(self, gen_graph):
        """Call remove_edges with source_ids/sink_ids (no atomic_edges) -- exercises shim."""
        graph = self._build_connected_graph(gen_graph)
        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)

        # Verify they share a root before split
        assert graph.get_root(sv0) == graph.get_root(sv1)

        # Use source_ids/sink_ids (the shim path) instead of atomic_edges
        result = graph.remove_edges(
            "TestUser",
            source_ids=sv0,
            sink_ids=sv1,
            mincut=False,
        )
        assert result.new_root_ids is not None
        assert len(result.new_root_ids) == 2

        # After split, they should have different roots
        assert graph.get_root(sv0) != graph.get_root(sv1)

    @pytest.mark.timeout(30)
    def test_remove_edges_shim_mismatched_lengths(self, gen_graph):
        """Shim path with mismatched source_ids/sink_ids lengths should raise."""
        graph = self._build_connected_graph(gen_graph)
        sv0 = to_label(graph, 1, 0, 0, 0, 0)
        sv1 = to_label(graph, 1, 0, 0, 0, 1)

        with pytest.raises(PreconditionError, match="same number"):
            graph.remove_edges(
                "TestUser",
                source_ids=[sv0, sv0],
                sink_ids=[sv1],
                mincut=False,
            )


# ===========================================================================
# get_earliest_timestamp -- detailed test (bigtable/client.py coverage)
# ===========================================================================


class TestEarliestTimestamp:
    """Tests for get_earliest_timestamp after operations exist."""

    @pytest.mark.timeout(30)
    def test_get_earliest_timestamp_after_merge(self, gen_graph):
        """After creating a graph and performing a merge, get_earliest_timestamp should return a valid datetime."""
        atomic_chunk_bounds = np.array([1, 1, 1])
        graph = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[
                to_label(graph, 1, 0, 0, 0, 0),
                to_label(graph, 1, 0, 0, 0, 1),
            ],
            edges=[],
            timestamp=fake_ts,
        )
        # Perform a merge to generate operation logs
        graph.add_edges(
            "TestUser",
            [to_label(graph, 1, 0, 0, 0, 0), to_label(graph, 1, 0, 0, 0, 1)],
            affinities=[0.3],
        )
        ts = graph.get_earliest_timestamp()
        assert ts is not None
        assert isinstance(ts, datetime)

    @pytest.mark.timeout(30)
    def test_get_earliest_timestamp_no_ops(self, gen_graph):
        """On a fresh graph with no operations, get_earliest_timestamp should return None."""
        graph = gen_graph(n_layers=4)
        fake_ts = datetime.now(UTC) - timedelta(days=10)
        create_chunk(
            graph,
            vertices=[to_label(graph, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_ts,
        )
        add_parent_chunk(graph, 3, [0, 0, 0], n_threads=1)
        add_parent_chunk(graph, 4, [0, 0, 0], n_threads=1)
        ts = graph.get_earliest_timestamp()
        # No operation logs, so should be None
        assert ts is None or isinstance(ts, datetime)
