"""Tests for pychunkedgraph.graph.sv_lookup.utils"""

from unittest.mock import MagicMock

import numpy as np

from pychunkedgraph.graph.sv_lookup import utils as sv_lookup_utils


class TestGetAtomicIdFromCoord:
    def test_exact_hit(self):
        """When the voxel at (x, y, z) contains an atomic ID whose root matches, return it."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        meta.cv.__getitem__ = MagicMock(return_value=np.array([[[42]]]))

        root_id = np.uint64(100)

        def fake_get_root(node_id, time_stamp=None):
            if node_id == 42:
                return root_id
            return root_id

        result = sv_lookup_utils.get_atomic_id_from_coord(
            meta, fake_get_root, 0, 0, 0, np.uint64(42), n_tries=1
        )
        assert result == np.uint64(42)

    def test_returns_none_when_no_match(self):
        """When no candidate atomic ID shares the same root, return None."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 0
        meta.cv.__getitem__ = MagicMock(return_value=np.array([[[0]]]))

        root_id = np.uint64(100)

        def fake_get_root(node_id, time_stamp=None):
            return root_id

        result = sv_lookup_utils.get_atomic_id_from_coord(
            meta, fake_get_root, 5, 5, 5, np.uint64(999), n_tries=1
        )
        assert result is None

    def test_mip_scaling(self):
        """Coordinates should be scaled by CV_MIP for x and y but not z."""
        meta = MagicMock()
        meta.data_source.CV_MIP = 2

        call_args = []

        def capture_getitem(self_mock, key):
            call_args.append(key)
            return np.array([[[7]]])

        meta.cv.__getitem__ = capture_getitem

        root_id = np.uint64(200)

        def fake_get_root(node_id, time_stamp=None):
            return root_id

        result = sv_lookup_utils.get_atomic_id_from_coord(
            meta, fake_get_root, 8, 12, 3, np.uint64(7), n_tries=1
        )
        assert result == np.uint64(7)
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
                return np.array([[[10]]])
            else:
                return np.array([[[10, 42]], [[10, 42]]])

        meta.cv.__getitem__ = expanding_getitem

        def fake_get_root(node_id, time_stamp=None):
            if node_id == 42:
                return target_root
            return wrong_root

        result = sv_lookup_utils.get_atomic_id_from_coord(
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
        meta.ocdbt_seg = False

        parent_id = np.uint64(42)
        coordinates = np.array([[10, 20, 30], [40, 50, 60]])

        def fake_get_roots(
            node_ids, time_stamp=None, stop_layer=None, fail_to_zero=False
        ):
            return np.array([parent_id] * len(node_ids), dtype=np.uint64)

        result = sv_lookup_utils.get_atomic_ids_from_coords(
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
        meta.ocdbt_seg = False

        parent_id = np.uint64(100)
        sv1 = np.uint64(10)
        sv2 = np.uint64(20)

        coordinates = np.array([[5, 5, 5], [6, 5, 5]])
        max_dist_nm = 150
        max_dist_vx = np.ceil(max_dist_nm / np.array([8, 8, 40])).astype(np.int32)

        bbox_min = np.min(coordinates, axis=0) - max_dist_vx
        bbox_max = np.max(coordinates, axis=0) + max_dist_vx + 1
        shape = bbox_max - bbox_min

        seg_block = np.zeros(tuple(shape), dtype=np.uint64)
        rel1 = coordinates[0] - bbox_min
        seg_block[rel1[0], rel1[1], rel1[2]] = sv1
        rel2 = coordinates[1] - bbox_min
        seg_block[rel2[0], rel2[1], rel2[2]] = sv2

        meta.cv.__getitem__ = MagicMock(return_value=seg_block)

        def fake_get_roots(
            node_ids, time_stamp=None, stop_layer=None, fail_to_zero=False
        ):
            result = []
            for nid in node_ids:
                if nid == sv1 or nid == sv2:
                    result.append(parent_id)
                else:
                    result.append(np.uint64(0))
            return np.array(result, dtype=np.uint64)

        result = sv_lookup_utils.get_atomic_ids_from_coords(
            meta,
            coordinates=coordinates,
            parent_id=parent_id,
            parent_id_layer=2,
            parent_ts=None,
            get_roots=fake_get_roots,
        )

        assert result is not None
        assert len(result) == 2
        assert np.uint64(result[0]) == sv1
        assert np.uint64(result[1]) == sv2
