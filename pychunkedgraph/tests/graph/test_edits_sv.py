"""Tests for pychunkedgraph.graph.edits_sv"""

import numpy as np
import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from pychunkedgraph.graph.edits_sv import (
    _coords_bbox,
    _voxel_crop,
    _parse_results,
    copy_parents_and_add_lineage,
    plan_sv_splits,
)
from pychunkedgraph.graph import attributes, basetypes


# ============================================================
# Tests: _voxel_crop
# ============================================================
class TestVoxelCrop:
    def test_no_overlap(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([10, 20, 30])
        bbe_ = np.array([20, 30, 40])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # No offset and no clipping
        assert crop == np.s_[0:None, 0:None, 0:None]

    def test_with_padding(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([9, 19, 29])
        bbe_ = np.array([21, 31, 41])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # Start offset = bbs - bbs_ = (1, 1, 1)
        # End offset: bbe_ - bbe = (1,1,1) != 0, so end = -1
        assert crop == np.s_[1:-1, 1:-1, 1:-1]

    def test_partial_padding(self):
        bbs = np.array([10, 20, 30])
        bbe = np.array([20, 30, 40])
        bbs_ = np.array([9, 20, 30])
        bbe_ = np.array([21, 30, 40])
        crop = _voxel_crop(bbs, bbe, bbs_, bbe_)
        # Only x has offset
        assert crop == np.s_[1:-1, 0:None, 0:None]


# ============================================================
# Tests: _parse_results
# ============================================================
class TestParseResults:
    def test_basic(self):
        seg = np.array([[[100, 100], [100, 200]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 2, 2])
        indices = np.array([[0, 0, 0], [0, 0, 1]])
        old_values = np.array([100, 100], dtype=basetypes.NODE_ID)
        new_values = np.array([300, 301], dtype=basetypes.NODE_ID)
        label_id_map = {1: np.uint64(300), 2: np.uint64(301)}
        results = [(indices, old_values, new_values, label_id_map)]

        updated_seg, old_new_map, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 300
        assert updated_seg[0, 0, 1] == 301
        assert 300 in old_new_map[100]
        assert 301 in old_new_map[100]
        assert new_id_label_map[np.uint64(300)] == 1
        assert new_id_label_map[np.uint64(301)] == 2

    def test_none_result_skipped(self):
        seg = np.array([[[100]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 1, 1])
        results = [None]
        updated_seg, old_new_map, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 100
        assert len(old_new_map) == 0
        assert len(new_id_label_map) == 0

    def test_multiple_results(self):
        seg = np.array([[[100, 200]]], dtype=basetypes.NODE_ID)
        bbs = np.array([0, 0, 0])
        bbe = np.array([1, 1, 2])
        result1 = (
            np.array([[0, 0, 0]]),
            np.array([100], dtype=basetypes.NODE_ID),
            np.array([300], dtype=basetypes.NODE_ID),
            {1: np.uint64(300)},
        )
        result2 = (
            np.array([[0, 0, 1]]),
            np.array([200], dtype=basetypes.NODE_ID),
            np.array([400], dtype=basetypes.NODE_ID),
            {1: np.uint64(400)},
        )
        results = [result1, result2]

        updated_seg, old_new_map, new_id_label_map = _parse_results(
            results, seg, bbs, bbe
        )
        assert updated_seg[0, 0, 0] == 300
        assert updated_seg[0, 0, 1] == 400
        assert 300 in old_new_map[100]
        assert 400 in old_new_map[200]


# ============================================================
# Tests: copy_parents_and_add_lineage
# ============================================================
class _FakeCell:
    """Mimics a bigtable cell with .value and .timestamp."""

    def __init__(self, value, timestamp=None):
        self.value = value
        self.timestamp = timestamp


class TestCopyParentsAndAddLineage:
    def _make_cg(self, parent_cells_map, children_cells_map=None):

        cg = MagicMock()
        cg.client.read_nodes.side_effect = lambda node_ids, properties: (
            parent_cells_map
            if properties is attributes.Hierarchy.Parent
            else (children_cells_map or {})
        )
        cg.client.mutate_row.side_effect = lambda key, val_dict, **kw: (
            key,
            val_dict,
            kw,
        )
        cg.cache.parents_cache = {}
        cg.cache.children_cache = {}
        return cg

    def test_single_old_to_two_new(self):
        """One old SV split into two new SVs. Each new SV gets the parent copied."""
        old = np.uint64(10)
        new1, new2 = np.uint64(101), np.uint64(102)
        parent = np.uint64(1000)

        parent_cells_map = {old: [_FakeCell(parent, timestamp=42)]}
        children_cells_map = {
            parent: [
                _FakeCell(
                    np.array([old, np.uint64(20)], dtype=basetypes.NODE_ID),
                    timestamp=42,
                )
            ]
        }
        cg = self._make_cg(parent_cells_map, children_cells_map)

        result = copy_parents_and_add_lineage(
            cg, operation_id=5, old_new_map={old: {new1, new2}}
        )

        # Should produce mutations:
        # - FormerIdentity + OperationID for each new SV (2)
        # - Parent copy for each new SV (2)
        # - NewIdentity on old SV (1)
        # - Updated children on parent (1)
        assert len(result) >= 5
        # Parent cache should have entries for both new SVs
        assert new1 in cg.cache.parents_cache or new2 in cg.cache.parents_cache
        # Children cache should replace old with new1, new2
        assert parent in cg.cache.children_cache
        children = cg.cache.children_cache[parent]
        assert new1 in children or int(new1) in children
        assert new2 in children or int(new2) in children

    def test_multiple_old_svs(self):
        """Two old SVs each split into new SVs, sharing the same parent."""
        old1, old2 = np.uint64(10), np.uint64(20)
        new1, new2, new3 = np.uint64(101), np.uint64(102), np.uint64(201)
        parent = np.uint64(1000)

        parent_cells_map = {
            old1: [_FakeCell(parent, timestamp=42)],
            old2: [_FakeCell(parent, timestamp=42)],
        }
        children_cells_map = {
            parent: [
                _FakeCell(
                    np.array([old1, old2, np.uint64(30)], dtype=basetypes.NODE_ID),
                    timestamp=42,
                )
            ]
        }
        cg = self._make_cg(parent_cells_map, children_cells_map)

        old_new_map = {old1: {new1, new2}, old2: {new3}}
        result = copy_parents_and_add_lineage(
            cg, operation_id=7, old_new_map=old_new_map
        )

        assert len(result) > 0
        # Children should replace old1 and old2 with new1, new2, new3, keep 30
        children = cg.cache.children_cache[parent]
        assert np.uint64(30) in children
        for nid in [new1, new2, new3]:
            assert nid in children or int(nid) in children

    def test_empty_old_new_map(self):
        """Empty map produces no mutations."""
        cg = self._make_cg({})
        result = copy_parents_and_add_lineage(cg, operation_id=1, old_new_map={})
        assert len(result) == 0

    def test_operation_id_stored(self):
        """Each new SV mutation includes the operation_id."""
        old = np.uint64(10)
        new1 = np.uint64(101)
        parent = np.uint64(1000)

        parent_cells_map = {old: [_FakeCell(parent, timestamp=1)]}
        children_cells_map = {
            parent: [_FakeCell(np.array([old], dtype=basetypes.NODE_ID), timestamp=1)]
        }
        cg = self._make_cg(parent_cells_map, children_cells_map)

        result = copy_parents_and_add_lineage(
            cg, operation_id=99, old_new_map={old: {new1}}
        )

        # Check that mutate_row was called with OperationID=99
        calls = cg.client.mutate_row.call_args_list
        op_id_found = False

        for call in calls:
            val_dict = call[0][1] if len(call[0]) > 1 else call[1].get("val_dict", {})
            if attributes.OperationLogs.OperationID in val_dict:
                assert val_dict[attributes.OperationLogs.OperationID] == 99
                op_id_found = True
        assert op_id_found

    def test_time_stamp_threaded_to_new_sv_writes(self):
        """New-SV writes (FormerIdentity/OperationID on new, NewIdentity
        on old) land at `time_stamp`. Parent-copy and Child-list writes
        preserve the old cell's timestamp so pre-op readers still see
        the old hierarchy.
        """
        from datetime import datetime, timezone

        old = np.uint64(10)
        new1 = np.uint64(101)
        parent = np.uint64(1000)

        old_cell_ts = 42  # old cell's timestamp, preserved on Parent/Child copies
        op_ts = datetime(2026, 4, 23, tzinfo=timezone.utc)  # op's logical write time

        parent_cells_map = {old: [_FakeCell(parent, timestamp=old_cell_ts)]}
        children_cells_map = {
            parent: [
                _FakeCell(
                    np.array([old], dtype=basetypes.NODE_ID), timestamp=old_cell_ts
                )
            ]
        }
        cg = self._make_cg(parent_cells_map, children_cells_map)

        copy_parents_and_add_lineage(
            cg, operation_id=7, old_new_map={old: {new1}}, time_stamp=op_ts
        )

        # Classify each mutate_row call by which column it writes.
        for call in cg.client.mutate_row.call_args_list:
            val_dict = call[0][1]
            kw = call[1]
            ts = kw.get("time_stamp")
            cols = set(val_dict.keys())

            if attributes.Hierarchy.FormerIdentity in cols:
                # New-SV lineage write — should use op's time_stamp.
                assert ts == op_ts, f"FormerIdentity write ts={ts}, expected {op_ts}"
            elif attributes.Hierarchy.NewIdentity in cols:
                # Old-SV NewIdentity write — should use op's time_stamp.
                assert ts == op_ts, f"NewIdentity write ts={ts}, expected {op_ts}"
            elif attributes.Hierarchy.Parent in cols:
                # Copied-parent write — preserves old cell's timestamp.
                assert (
                    ts == old_cell_ts
                ), f"Parent-copy write ts={ts}, expected {old_cell_ts}"
            elif attributes.Hierarchy.Child in cols:
                # Updated-children write on L2 parent — preserves old timestamp.
                assert (
                    ts == old_cell_ts
                ), f"Child-list write ts={ts}, expected {old_cell_ts}"


# ============================================================
# Tests: _coords_bbox / plan_sv_splits bbox is seed-driven, not rep-driven
# ============================================================
class TestCoordsBbox:
    def _make_cg(self, chunk_size=(64, 64, 64), volume=(1024, 1024, 1024)):
        cg = MagicMock()
        cg.meta.graph_config.CHUNK_SIZE = list(chunk_size)
        cg.meta.voxel_bounds = np.array(
            [[0, volume[0]], [0, volume[1]], [0, volume[2]]]
        )
        cg.get_chunk_id.side_effect = lambda layer, x, y, z: (
            (layer << 60) | (x << 40) | (y << 20) | z
        )
        return cg

    def test_envelope_around_seeds_with_one_chunk_margin(self):
        cg = self._make_cg(chunk_size=(64, 64, 64))
        src = np.array([[100, 200, 300]])
        sink = np.array([[150, 250, 350]])
        bbs, bbe = _coords_bbox(cg, src, sink)
        # min - chunk_size, max + chunk_size, clipped to volume bounds.
        np.testing.assert_array_equal(bbs, np.array([100 - 64, 200 - 64, 300 - 64]))
        np.testing.assert_array_equal(bbe, np.array([150 + 64, 250 + 64, 350 + 64]))

    def test_clipped_to_volume_bounds(self):
        cg = self._make_cg(chunk_size=(64, 64, 64), volume=(256, 256, 256))
        src = np.array([[10, 10, 10]])
        sink = np.array([[250, 250, 250]])
        bbs, bbe = _coords_bbox(cg, src, sink)
        # Lower seed - 64 = -54 → clipped to 0; upper seed + 64 = 314 → clipped to 256.
        np.testing.assert_array_equal(bbs, np.array([0, 0, 0]))
        np.testing.assert_array_equal(bbe, np.array([256, 256, 256]))

    def test_plan_sv_splits_bbox_independent_of_rep_extent(self):
        """The returned per-task bbox follows the seeds, not the rep's
        cross-chunk pieces. A rep whose pieces span the whole volume
        produces the same tight bbox as a rep with one piece, given the
        same src/sink coords.
        """
        cg = self._make_cg(chunk_size=(64, 64, 64), volume=(1024, 1024, 1024))

        # Two source/sink IDs that map to the same rep — the SV-split
        # trigger condition. The rep's other pieces (b..z) sit far from
        # the seeds. They would have ballooned the old `_rep_bbox`; the
        # new `_coords_bbox` ignores them.
        rep = np.uint64(1)
        sv_remapping = {
            np.uint64(10): rep,  # src
            np.uint64(20): rep,  # sink
            **{np.uint64(100 + i): rep for i in range(28)},  # 28 distant pieces
        }

        source_ids = np.array([10], dtype=basetypes.NODE_ID)
        sink_ids = np.array([20], dtype=basetypes.NODE_ID)
        source_coords = np.array([[100, 200, 300]])
        sink_coords = np.array([[150, 250, 350]])

        tasks, _ = plan_sv_splits(
            cg,
            sv_remapping=sv_remapping,
            source_ids=source_ids,
            sink_ids=sink_ids,
            source_coords=source_coords,
            sink_coords=sink_coords,
        )
        assert len(tasks) == 1
        np.testing.assert_array_equal(
            tasks[0].bbs, np.array([100 - 64, 200 - 64, 300 - 64])
        )
        np.testing.assert_array_equal(
            tasks[0].bbe, np.array([150 + 64, 250 + 64, 350 + 64])
        )
