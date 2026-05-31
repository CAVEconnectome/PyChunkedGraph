"""Tests for pychunkedgraph.scheduling.partition (generic LPT partitioning)."""

from pychunkedgraph.scheduling import WorkItem, lpt_partition, n_bins_for


def _loads(bins):
    return [sum(it.weight for it in b) for b in bins]


class TestLptPartition:
    def test_places_every_item_once(self):
        items = [WorkItem(payload=i, weight=(i % 7) + 1) for i in range(100)]
        bins = lpt_partition(items, 8)
        placed = sorted(it.payload for b in bins for it in b)
        assert placed == list(range(100))

    def test_uniform_weights_split_evenly(self):
        bins = lpt_partition([WorkItem(i, 1) for i in range(100)], 10)
        loads = _loads(bins)
        assert max(loads) - min(loads) <= 1

    def test_bins_returned_heaviest_first(self):
        items = [WorkItem(i, w) for i, w in enumerate([5, 1, 9, 3, 7, 2, 8])]
        loads = _loads(lpt_partition(items, 3))
        assert loads == sorted(loads, reverse=True)

    def test_makespan_within_four_thirds_of_optimal(self):
        weights = [50, 40, 30, 20] + [2] * 200
        items = [WorkItem(i, w) for i, w in enumerate(weights)]
        bins = lpt_partition(items, 16)
        lower_bound = max(max(weights), sum(weights) / len(bins))
        assert _loads(bins)[0] <= (4 / 3) * lower_bound + 1e-9

    def test_dominant_item_isolated(self):
        # one giant + many tiny: the giant's bin should carry essentially just it,
        # not strand a worker behind a count-balanced batch of hangers-on.
        items = [WorkItem("giant", 1000)] + [WorkItem(i, 1) for i in range(500)]
        bins = lpt_partition(items, n_bins_for(len(items), target_per_bin=32))
        heaviest = bins[0]
        assert any(it.payload == "giant" for it in heaviest)
        assert _loads(bins)[0] <= 1000 + 5  # at most a few unit hangers-on

    def test_empty_input_yields_no_bins(self):
        assert lpt_partition([], 8) == []

    def test_single_item_one_bin(self):
        bins = lpt_partition([WorkItem("x", 5)], 8)
        assert len(bins) == 1 and bins[0][0].payload == "x"

    def test_more_bins_than_items_clamped_no_empties(self):
        bins = lpt_partition([WorkItem(i, 1) for i in range(3)], 10)
        assert len(bins) == 3
        assert all(b for b in bins)

    def test_default_weight_is_one(self):
        assert WorkItem(payload="a").weight == 1.0


class TestNBinsFor:
    def test_ceil_division(self):
        assert n_bins_for(100, 32) == 4
        assert n_bins_for(64, 32) == 2

    def test_at_least_one(self):
        assert n_bins_for(0, 32) == 1
        assert n_bins_for(5, 0) == 5  # target clamped to >=1
