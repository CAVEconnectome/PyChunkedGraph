"""Tests for pychunkedgraph.graph.utils.id_helpers"""

from unittest.mock import MagicMock

import numpy as np

from pychunkedgraph.graph.utils import id_helpers
from pychunkedgraph.graph.chunks import utils as chunk_utils

from .helpers import to_label


class TestGetSegmentIdLimit:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = to_label(graph, 1, 0, 0, 0, 1)
        limit = id_helpers.get_segment_id_limit(graph.meta, node_id)
        assert limit > 0
        assert isinstance(limit, np.uint64)


class TestGetSegmentId:
    def test_basic(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = to_label(graph, 1, 0, 0, 0, 42)
        seg_id = id_helpers.get_segment_id(graph.meta, node_id)
        assert seg_id == 42


class TestGetNodeId:
    def test_from_chunk_id(self, gen_graph):
        graph = gen_graph(n_layers=4)
        chunk_id = chunk_utils.get_chunk_id(graph.meta, layer=1, x=0, y=0, z=0)
        node_id = id_helpers.get_node_id(
            graph.meta, segment_id=np.uint64(5), chunk_id=chunk_id
        )
        assert id_helpers.get_segment_id(graph.meta, node_id) == 5
        assert chunk_utils.get_chunk_layer(graph.meta, node_id) == 1

    def test_from_components(self, gen_graph):
        graph = gen_graph(n_layers=4)
        node_id = id_helpers.get_node_id(
            graph.meta, segment_id=np.uint64(7), layer=2, x=1, y=2, z=3
        )
        assert id_helpers.get_segment_id(graph.meta, node_id) == 7
        assert chunk_utils.get_chunk_layer(graph.meta, node_id) == 2
        coords = chunk_utils.get_chunk_coordinates(graph.meta, node_id)
        np.testing.assert_array_equal(coords, [1, 2, 3])


class TestGetAtomicIdFromCoord:
    def test_exact_hit(self):
        """When the voxel at (x, y, z) contains an atomic ID whose root matches, return it."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        # meta.cv[x_l:x_h, y_l:y_h, z_l:z_h] returns an array block.
        # For i_try=0: x_l = x - (-1)^2 = x-1, but clamped to 0 if negative;
        # x_h = x + 1 + (-1)^2 = x+2. With x=0: x_l=0, x_h=2, etc.
        # Simplest: put target atomic_id=42 everywhere in a small block.
        meta.cv.__getitem__ = MagicMock(return_value=np.array([[[42]]]))

        root_id = np.uint64(100)

        def fake_get_root(node_id, time_stamp=None):
            if node_id == 42:
                return root_id
            return root_id  # same root for all

        result = id_helpers.get_atomic_id_from_coord(
            meta, fake_get_root, 0, 0, 0, np.uint64(42), n_tries=1
        )
        assert result == np.uint64(42)

    def test_returns_none_when_no_match(self):
        """When no candidate atomic ID shares the same root, return None."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        # Return only zeros (background) from cloudvolume
        meta.cv.__getitem__ = MagicMock(return_value=np.array([[[0]]]))

        root_id = np.uint64(100)

        def fake_get_root(node_id, time_stamp=None):
            return root_id

        result = id_helpers.get_atomic_id_from_coord(
            meta, fake_get_root, 5, 5, 5, np.uint64(999), n_tries=1
        )
        # Only candidate is 0, which is skipped, so result should be None
        assert result is None

    def test_mip_scaling(self):
        """Coordinates should be scaled by CV_MIP for x and y but not z."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 2  # scale factor of 4 for x,y

        call_args = []

        def capture_getitem(self_mock, key):
            call_args.append(key)
            return np.array([[[7]]])

        meta.cv.__getitem__ = capture_getitem

        root_id = np.uint64(200)

        def fake_get_root(node_id, time_stamp=None):
            return root_id

        result = id_helpers.get_atomic_id_from_coord(
            meta, fake_get_root, 8, 12, 3, np.uint64(7), n_tries=1
        )
        assert result == np.uint64(7)
        # Verify that the function was called (coordinates are scaled)
        assert len(call_args) >= 1

    def test_retry_expands_search(self):
        """With multiple tries, the search area should expand to find a matching ID."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0

        target_root = np.uint64(500)
        wrong_root = np.uint64(999)
        call_count = [0]

        def expanding_getitem(self_mock, key):
            call_count[0] += 1
            if call_count[0] == 1:
                # First try returns a non-matching ID
                return np.array([[[10]]])
            else:
                # Second try returns the matching ID
                return np.array([[[10, 42]], [[10, 42]]])

        meta.cv.__getitem__ = expanding_getitem

        def fake_get_root(node_id, time_stamp=None):
            if node_id == 42:
                return target_root
            return wrong_root

        # parent_id=42 -> root=500; candidates: try1 has only 10 (root=999), try2 has 42 (root=500)
        result = id_helpers.get_atomic_id_from_coord(
            meta, fake_get_root, 5, 5, 5, np.uint64(42), n_tries=3
        )
        assert result == np.uint64(42)
        assert call_count[0] >= 2


class TestGetAtomicIdsFromCoords:
    def test_layer1_returns_parent_id(self):
        """When parent_id is already layer 1, return parent_id for all coordinates."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        meta.resolution = np.array([1, 1, 1])

        parent_id = np.uint64(42)
        coordinates = np.array([[10, 20, 30], [40, 50, 60]])

        def fake_get_roots(
            node_ids, time_stamp=None, stop_layer=None, fail_to_zero=False
        ):
            return np.array([parent_id] * len(node_ids), dtype=np.uint64)

        result = id_helpers.get_atomic_ids_from_coords(
            meta,
            coordinates=coordinates,
            parent_id=parent_id,
            parent_id_layer=1,
            parent_ts=None,
            get_roots=fake_get_roots,
        )

        np.testing.assert_array_equal(result, [parent_id, parent_id])

    def test_higher_layer_with_mock_cv(self):
        """Test with a mocked CloudVolume that returns a known segmentation block."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        meta.resolution = np.array([8, 8, 40])

        parent_id = np.uint64(100)
        sv1 = np.uint64(10)
        sv2 = np.uint64(20)

        # Create a small segmentation volume (the CV mock)
        # Coordinates: two points at [5, 5, 5] and [6, 5, 5]
        coordinates = np.array([[5, 5, 5], [6, 5, 5]])
        max_dist_nm = 150
        max_dist_vx = np.ceil(max_dist_nm / np.array([8, 8, 40])).astype(np.int32)

        # Build a segmentation block big enough for the bounding box
        bbox_min = np.min(coordinates, axis=0) - max_dist_vx
        bbox_max = np.max(coordinates, axis=0) + max_dist_vx + 1
        shape = bbox_max - bbox_min

        seg_block = np.zeros(tuple(shape), dtype=np.uint64)
        # Place sv1 at relative position of coordinate [5,5,5]
        rel1 = coordinates[0] - bbox_min
        seg_block[rel1[0], rel1[1], rel1[2]] = sv1
        # Place sv2 at relative position of coordinate [6,5,5]
        rel2 = coordinates[1] - bbox_min
        seg_block[rel2[0], rel2[1], rel2[2]] = sv2

        meta.cv.__getitem__ = MagicMock(return_value=seg_block)

        def fake_get_roots(
            node_ids, time_stamp=None, stop_layer=None, fail_to_zero=False
        ):
            # Map sv1 and sv2 to parent_id, everything else to 0
            result = []
            for nid in node_ids:
                if nid == sv1 or nid == sv2:
                    result.append(parent_id)
                else:
                    result.append(np.uint64(0))
            return np.array(result, dtype=np.uint64)

        result = id_helpers.get_atomic_ids_from_coords(
            meta,
            coordinates=coordinates,
            parent_id=parent_id,
            parent_id_layer=2,
            parent_ts=None,
            get_roots=fake_get_roots,
        )

        assert result is not None
        assert len(result) == 2
        # Each coordinate should map to one of our supervoxels
        assert np.uint64(result[0]) == sv1
        assert np.uint64(result[1]) == sv2
