"""Tests for pychunkedgraph.graph.utils.generic"""

import datetime

import numpy as np
import pytz
import pytest

from pychunkedgraph.graph.utils.generic import (
    compute_indices_pandas,
    log_n,
    compute_bitmasks,
    get_max_time,
    get_min_time,
    time_min,
    get_valid_timestamp,
    get_bounding_box,
    filter_failed_node_ids,
    _get_google_compatible_time_stamp,
    mask_nodes_by_bounding_box,
    get_parents_at_timestamp,
)


class TestLogN:
    def test_base2(self):
        assert log_n(8, 2) == pytest.approx(3.0)

    def test_base10(self):
        assert log_n(1000, 10) == pytest.approx(3.0)

    def test_other_base(self):
        assert log_n(27, 3) == pytest.approx(3.0)

    def test_array_input(self):
        result = log_n(np.array([4, 8, 16]), 2)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])


class TestComputeBitmasks:
    def test_basic(self):
        bm = compute_bitmasks(4)
        assert 1 in bm
        assert 2 in bm
        assert 3 in bm
        assert 4 in bm

    def test_layer_1_equals_layer_2(self):
        bm = compute_bitmasks(5)
        assert bm[1] == bm[2]

    def test_insufficient_bits_raises(self):
        with pytest.raises(ValueError, match="not enough"):
            compute_bitmasks(4, s_bits_atomic_layer=0)


class TestTimeFunctions:
    def test_get_max_time(self):
        t = get_max_time()
        assert isinstance(t, datetime.datetime)
        assert t.year == 9999

    def test_get_min_time(self):
        t = get_min_time()
        assert isinstance(t, datetime.datetime)
        assert t.year == 2000

    def test_time_min(self):
        assert time_min() == get_min_time()


class TestGetValidTimestamp:
    def test_none_returns_utc_now(self):
        before = datetime.datetime.now(datetime.timezone.utc)
        result = get_valid_timestamp(None)
        after = datetime.datetime.now(datetime.timezone.utc)
        assert result.tzinfo is not None
        # get_valid_timestamp rounds down to millisecond precision,
        # so result may be slightly before `before`
        tolerance = datetime.timedelta(milliseconds=1)
        assert before - tolerance <= result <= after

    def test_naive_gets_localized(self):
        naive = datetime.datetime(2023, 6, 15, 12, 0, 0)
        result = get_valid_timestamp(naive)
        assert result.tzinfo is not None

    def test_aware_passthrough(self):
        aware = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        result = get_valid_timestamp(aware)
        assert result.tzinfo is not None


class TestGoogleCompatibleTimestamp:
    def test_round_down(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 1500)
        result = _get_google_compatible_time_stamp(ts, round_up=False)
        assert result.microsecond % 1000 == 0
        assert result.microsecond == 1000

    def test_round_up(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 1500)
        result = _get_google_compatible_time_stamp(ts, round_up=True)
        assert result.microsecond % 1000 == 0
        assert result.microsecond == 2000

    def test_exact_no_change(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 3000)
        result = _get_google_compatible_time_stamp(ts)
        assert result == ts


class TestGetBoundingBox:
    def test_normal(self):
        source = np.array([[10, 20, 30]])
        sink = np.array([[50, 60, 70]])
        bbox = get_bounding_box(source, sink, bb_offset=(5, 5, 5))
        np.testing.assert_array_equal(bbox[0], [5, 15, 25])
        np.testing.assert_array_equal(bbox[1], [55, 65, 75])

    def test_none_coords(self):
        assert get_bounding_box(None, [[1, 2, 3]]) is None
        assert get_bounding_box([[1, 2, 3]], None) is None


class TestFilterFailedNodeIds:
    def test_basic(self):
        row_ids = np.array([10, 20, 30, 40], dtype=np.uint64)
        segment_ids = np.array([4, 3, 2, 1], dtype=np.uint64)
        max_children_ids = np.array([100, 100, 200, 200])
        result = filter_failed_node_ids(row_ids, segment_ids, max_children_ids)
        # Only the first occurrence of each max_children_id (by descending segment_id) survives
        assert len(result) == 2


class TestMaskNodesByBoundingBox:
    def test_none_bbox(self):
        nodes = np.array([1, 2, 3], dtype=np.uint64)
        result = mask_nodes_by_bounding_box(None, nodes, bounding_box=None)
        assert np.all(result)


class TestGetParentsAtTimestamp:
    def test_normal_lookup(self):
        ts1 = datetime.datetime(2023, 1, 1)
        ts2 = datetime.datetime(2023, 6, 1)
        ts_map = {
            10: {ts2: 100, ts1: 50},
        }
        parents, skipped = get_parents_at_timestamp([10], ts_map, ts2)
        assert 100 in parents
        assert len(skipped) == 0

    def test_missing_key(self):
        parents, skipped = get_parents_at_timestamp([99], {}, datetime.datetime.now())
        assert len(parents) == 0
        assert 99 in skipped

    def test_unique(self):
        ts = datetime.datetime(2023, 6, 1)
        ts_map = {
            10: {ts: 100},
            20: {ts: 100},
        }
        parents, _ = get_parents_at_timestamp([10, 20], ts_map, ts, unique=True)
        assert len(parents) == 1


class TestComputeIndicesPandas:
    def test_basic(self):
        data = np.array([1, 2, 1, 2, 3])
        result = compute_indices_pandas(data)
        assert 1 in result.index
        assert 2 in result.index
        assert 3 in result.index
