"""Integration tests for stale edge detection and resolution.

Tests get_stale_nodes() and get_new_nodes() from stale.py using real graph
operations through the BigTable emulator.
"""

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from .helpers import create_chunk, to_label
from ..graph.edges.stale import get_stale_nodes, get_new_nodes
from ..ingest.create.parent_layer import add_parent_chunk


class TestStaleEdges:
    @pytest.mark.timeout(30)
    def test_stale_nodes_detected_after_split(self, gen_graph):
        """
        After a split, the old L2 parent IDs become stale.
        get_stale_nodes should identify them.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Get old parents before edit
        old_root = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )

        # The old root should now be stale
        stale = get_stale_nodes(cg, [old_root])
        assert old_root in stale

    @pytest.mark.timeout(30)
    def test_no_stale_nodes_for_current_ids(self, gen_graph):
        """
        Current (post-edit) node IDs should not be flagged as stale.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )

        # Current roots should not be stale
        new_root_1 = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        new_root_2 = cg.get_root(to_label(cg, 1, 1, 0, 0, 0))
        stale = get_stale_nodes(cg, [new_root_1, new_root_2])
        assert new_root_1 not in stale
        assert new_root_2 not in stale

    @pytest.mark.timeout(30)
    def test_get_new_nodes_resolves_to_correct_layer(self, gen_graph):
        """
        get_new_nodes should follow the parent chain from a supervoxel
        to the correct layer and return the current node at that layer.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Get L2 parent of SV 1 before edit
        sv1 = to_label(cg, 1, 0, 0, 0, 0)
        old_l2_parent = cg.get_parent(sv1)

        # Split
        cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )

        # get_new_nodes should resolve SV to its current L2 parent
        new_l2 = get_new_nodes(cg, np.array([sv1], dtype=np.uint64), layer=2)
        current_l2_parent = cg.get_parent(sv1)
        assert new_l2[0] == current_l2_parent

    @pytest.mark.timeout(30)
    def test_no_stale_nodes_in_unaffected_region(self, gen_graph):
        """
        Nodes not involved in an edit should not be flagged as stale.

        ┌─────┬─────┬─────┐
        │  A¹ │  B¹ │  C¹ │
        │  1━━┿━━2  │  3  │
        │     │     │     │
        └─────┴─────┴─────┘
        """
        cg = gen_graph(n_layers=4)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        # Chunk C - isolated node, not connected to A or B
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 2, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )

        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 3, [1, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        # Get the isolated node's root before edit
        isolated_root = cg.get_root(to_label(cg, 1, 2, 0, 0, 0))

        # Split nodes 1 and 2
        cg.remove_edges(
            "test_user",
            source_ids=to_label(cg, 1, 0, 0, 0, 0),
            sink_ids=to_label(cg, 1, 1, 0, 0, 0),
            mincut=False,
        )

        # The isolated root should not be stale — it was unaffected
        stale = get_stale_nodes(cg, [isolated_root])
        assert isolated_root not in stale

    @pytest.mark.timeout(30)
    def test_get_new_nodes_returns_self_for_non_stale(self, gen_graph):
        """
        For freshly created nodes with no edits, get_new_nodes should return
        the nodes themselves (identity mapping).

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg = gen_graph(n_layers=4)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        sv = to_label(cg, 1, 0, 0, 0, 0)
        l2_parent = cg.get_parent(sv)

        # get_new_nodes at layer 2 should return the same L2 parent
        result = get_new_nodes(cg, np.array([sv], dtype=np.uint64), layer=2)
        assert result[0] == l2_parent

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_empty_for_fresh_graph(self, gen_graph):
        """
        In a freshly built graph with no edits, no nodes should be stale.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        root = cg.get_root(to_label(cg, 1, 0, 0, 0, 0))
        l2_0 = cg.get_parent(to_label(cg, 1, 0, 0, 0, 0))
        l2_1 = cg.get_parent(to_label(cg, 1, 1, 0, 0, 0))

        # No edits have been performed, so all nodes should be non-stale
        stale = get_stale_nodes(cg, [root, l2_0, l2_1])
        assert len(stale) == 0

    @pytest.mark.timeout(30)
    def test_get_new_nodes_multiple_svs(self, gen_graph):
        """
        get_new_nodes with multiple supervoxels should return an array
        of the same length, each mapped to its current L2 parent.

        ┌─────┬─────┐
        │  A¹ │  B¹ │
        │  1━━┿━━2  │
        │     │     │
        └─────┴─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[(to_label(cg, 1, 0, 0, 0, 0), to_label(cg, 1, 1, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 1, 0, 0, 0)],
            edges=[(to_label(cg, 1, 1, 0, 0, 0), to_label(cg, 1, 0, 0, 0, 0), 0.5)],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        sv1 = to_label(cg, 1, 0, 0, 0, 0)
        sv2 = to_label(cg, 1, 1, 0, 0, 0)
        svs = np.array([sv1, sv2], dtype=np.uint64)

        result = get_new_nodes(cg, svs, layer=2)
        assert result.shape == (2,)
        # Each SV should map to its L2 parent
        assert result[0] == cg.get_parent(sv1)
        assert result[1] == cg.get_parent(sv2)

    @pytest.mark.timeout(30)
    def test_get_new_nodes_with_duplicate_svs(self, gen_graph):
        """
        get_new_nodes should handle duplicate SVs correctly,
        returning the same result for duplicate inputs.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg = gen_graph(n_layers=3)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        create_chunk(
            cg,
            vertices=[to_label(cg, 1, 0, 0, 0, 0)],
            edges=[],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        sv = to_label(cg, 1, 0, 0, 0, 0)
        svs = np.array([sv, sv, sv], dtype=np.uint64)

        result = get_new_nodes(cg, svs, layer=2)
        assert result.shape == (3,)
        # All should map to the same L2 parent
        expected = cg.get_parent(sv)
        assert np.all(result == expected)

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_with_l2_ids_after_merge(self, gen_graph):
        """
        After a merge, the old L2 IDs should become stale.

        ┌─────┐
        │  A¹ │
        │ 1 2 │  (isolated, then merged)
        │     │
        └─────┘
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        cg = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        sv1 = to_label(cg, 1, 0, 0, 0, 1)

        create_chunk(
            cg,
            vertices=[sv0, sv1],
            edges=[],
            timestamp=fake_timestamp,
        )

        # Get L2 parents before merge (each SV has its own L2 parent)
        old_l2_0 = cg.get_parent(sv0)
        old_l2_1 = cg.get_parent(sv1)

        # Merge
        cg.add_edges(
            "test_user",
            [sv0, sv1],
            affinities=[0.3],
        )

        # Old L2 parents should now be stale
        stale = get_stale_nodes(cg, [old_l2_0, old_l2_1])
        assert old_l2_0 in stale or old_l2_1 in stale

    @pytest.mark.timeout(30)
    def test_get_stale_nodes_returns_numpy_array(self, gen_graph):
        """
        get_stale_nodes should always return a numpy ndarray, even when
        no nodes are stale.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        atomic_chunk_bounds = np.array([1, 1, 1])
        cg = gen_graph(n_layers=2, atomic_chunk_bounds=atomic_chunk_bounds)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        sv0 = to_label(cg, 1, 0, 0, 0, 0)
        create_chunk(
            cg,
            vertices=[sv0],
            edges=[],
            timestamp=fake_timestamp,
        )

        root = cg.get_root(sv0)
        stale = get_stale_nodes(cg, [root])
        assert isinstance(stale, np.ndarray)

    @pytest.mark.timeout(30)
    def test_get_new_nodes_at_root_layer(self, gen_graph):
        """
        get_new_nodes called with layer=root_layer should return the root node.

        ┌─────┐
        │  A¹ │
        │  1  │
        │     │
        └─────┘
        """
        cg = gen_graph(n_layers=4)
        fake_timestamp = datetime.now(UTC) - timedelta(days=10)

        sv = to_label(cg, 1, 0, 0, 0, 0)
        create_chunk(
            cg,
            vertices=[sv],
            edges=[],
            timestamp=fake_timestamp,
        )
        add_parent_chunk(cg, 3, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)
        add_parent_chunk(cg, 4, [0, 0, 0], time_stamp=fake_timestamp, n_threads=1)

        root = cg.get_root(sv)
        root_layer = cg.get_chunk_layer(root)

        result = get_new_nodes(cg, np.array([sv], dtype=np.uint64), layer=root_layer)
        assert result.shape == (1,)
        assert result[0] == root
