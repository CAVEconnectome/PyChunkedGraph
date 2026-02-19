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
