"""Characterization tests for combining ``Edges`` instances.

These pin the behavior of folding with ``+`` (``Edges.__add__``), which is how
``ChunkedGraph.get_l2_agglomerations`` combines per-chunk edge sets:

    all_chunk_edges = reduce(
        lambda x, y: x + y, chain(edges_d.values(), fake_edges.values()), Edges([], [])
    )

Every step of that fold reallocates all four arrays, so it is a candidate for
replacement by a single bulk concatenation. Nothing in the suite covered it before:
the tests that reach ``get_l2_agglomerations`` set ``mock_edges``, which both skips
the chunk read (leaving the fold with an empty sequence) and discards the fold's
result. These tests exist so that such a replacement is verifiable -- they describe
the behavior any bulk implementation has to reproduce, not any particular one.
"""

from functools import reduce
from itertools import chain

import numpy as np
import pytest

from ..graph.edges import Edges, in_sorted
from ..graph.utils import basetypes


def _edges(start, count, *, with_attrs=True):
    """Build a deterministic Edges of ``count`` edges, ids offset by ``start``."""
    node_ids1 = np.arange(start, start + count, dtype=basetypes.NODE_ID)
    node_ids2 = np.arange(start + 1000, start + 1000 + count, dtype=basetypes.NODE_ID)
    if not with_attrs:
        return Edges(node_ids1, node_ids2)
    return Edges(
        node_ids1,
        node_ids2,
        affinities=np.arange(count, dtype=basetypes.EDGE_AFFINITY) + 0.5,
        areas=np.arange(count, dtype=basetypes.EDGE_AREA) + 3,
    )


def _fold(parts):
    """The exact fold used by get_l2_agglomerations."""
    return reduce(lambda x, y: x + y, chain(parts), Edges([], []))


def _bulk(parts):
    """Reference bulk concatenation: one np.concatenate per attribute."""
    return Edges(
        np.concatenate([p.node_ids1 for p in parts] or [np.array([], dtype=basetypes.NODE_ID)]),
        np.concatenate([p.node_ids2 for p in parts] or [np.array([], dtype=basetypes.NODE_ID)]),
        affinities=np.concatenate(
            [p.affinities for p in parts] or [np.array([], dtype=basetypes.EDGE_AFFINITY)]
        ),
        areas=np.concatenate(
            [p.areas for p in parts] or [np.array([], dtype=basetypes.EDGE_AREA)]
        ),
    )


def _assert_same(actual, expected):
    np.testing.assert_array_equal(actual.node_ids1, expected.node_ids1)
    np.testing.assert_array_equal(actual.node_ids2, expected.node_ids2)
    np.testing.assert_array_equal(actual.affinities, expected.affinities)
    np.testing.assert_array_equal(actual.areas, expected.areas)


class TestEdgesConcatenation:
    def test_add_combines_all_four_arrays(self):
        """``+`` must carry affinities and areas, not just the node ids."""
        a, b = _edges(0, 3), _edges(100, 2)
        combined = a + b

        assert len(combined) == 5
        np.testing.assert_array_equal(
            combined.node_ids1, np.concatenate([a.node_ids1, b.node_ids1])
        )
        np.testing.assert_array_equal(
            combined.node_ids2, np.concatenate([a.node_ids2, b.node_ids2])
        )
        np.testing.assert_array_equal(
            combined.affinities, np.concatenate([a.affinities, b.affinities])
        )
        np.testing.assert_array_equal(combined.areas, np.concatenate([a.areas, b.areas]))

    def test_add_preserves_order(self):
        """Order is positional: consumers zip edges against affinities/areas."""
        a, b = _edges(0, 2), _edges(100, 2)

        assert (a + b).node_ids1.tolist() == a.node_ids1.tolist() + b.node_ids1.tolist()
        assert (b + a).node_ids1.tolist() == b.node_ids1.tolist() + a.node_ids1.tolist()

    def test_add_leaves_operands_unmodified(self):
        a, b = _edges(0, 3), _edges(100, 2)
        before = a.node_ids1.copy()

        _ = a + b

        np.testing.assert_array_equal(a.node_ids1, before)
        assert len(a) == 3 and len(b) == 2

    def test_fold_matches_bulk_concatenation(self):
        """The invariant a bulk replacement has to satisfy."""
        parts = [_edges(i * 100, i + 1) for i in range(6)]

        _assert_same(_fold(parts), _bulk(parts))
        assert len(_fold(parts)) == sum(len(p) for p in parts)

    def test_fold_of_empty_sequence(self):
        """get_l2_agglomerations folds an empty chain whenever mock_edges is set."""
        folded = _fold([])

        assert len(folded) == 0
        assert folded.node_ids1.size == 0
        assert folded.get_pairs().shape == (0, 2)

    def test_fold_of_single_element(self):
        part = _edges(0, 4)

        _assert_same(_fold([part]), part)

    def test_fold_with_empty_parts_interleaved(self):
        """Chunks with no edges are common; they must not perturb the result."""
        parts = [_edges(0, 2), Edges([], []), _edges(100, 3), Edges([], [])]

        _assert_same(_fold(parts), _bulk([p for p in parts if len(p)]))
        assert len(_fold(parts)) == 5

    def test_fold_materializes_defaults_for_parts_without_attrs(self):
        """Edges built without affinities/areas still contribute full arrays."""
        with_attrs, without = _edges(0, 2), _edges(100, 3, with_attrs=False)

        folded = _fold([with_attrs, without])

        assert folded.affinities.size == len(folded)
        assert folded.areas.size == len(folded)
        np.testing.assert_array_equal(folded.affinities[:2], with_attrs.affinities)
        np.testing.assert_array_equal(folded.affinities[2:], without.affinities)

    def test_fold_preserves_dtypes(self):
        """Downstream code indexes these as id/affinity/area types."""
        folded = _fold([_edges(0, 2), _edges(100, 3)])

        assert folded.node_ids1.dtype == basetypes.NODE_ID
        assert folded.node_ids2.dtype == basetypes.NODE_ID
        assert folded.affinities.dtype == _edges(0, 1).affinities.dtype
        assert folded.areas.dtype == _edges(0, 1).areas.dtype

    def test_get_pairs_after_fold(self):
        """get_l2_agglomerations passes the folded result on as pairs."""
        parts = [_edges(0, 2), _edges(100, 3)]

        pairs = _fold(parts).get_pairs()

        assert pairs.shape == (5, 2)
        np.testing.assert_array_equal(pairs[:, 0], _bulk(parts).node_ids1)
        np.testing.assert_array_equal(pairs[:, 1], _bulk(parts).node_ids2)

    @pytest.mark.parametrize("count", [0, 1, 2, 10])
    def test_fold_matches_bulk_for_various_lengths(self, count):
        parts = [_edges(i * 100, 2) for i in range(count)]

        folded = _fold(parts)

        assert len(folded) == 2 * count
        if count:
            _assert_same(folded, _bulk(parts))


class TestEdgesConcatenateReplacesFold:
    """Edges.concatenate replaced the reduce in get_l2_agglomerations.

    Every case above that pins the fold is re-asserted here against concatenate, so the
    two are interchangeable. If they ever diverge these fail rather than the change
    silently altering what get_l2_agglomerations hands to categorize_edges_v2.
    """

    @pytest.mark.parametrize("count", [0, 1, 2, 3, 10])
    def test_matches_fold_for_various_lengths(self, count):
        parts = [_edges(i * 100, i + 1) for i in range(count)]

        _assert_same(Edges.concatenate(parts), _fold(parts))

    def test_matches_fold_with_empty_parts_interleaved(self):
        parts = [_edges(0, 2), Edges([], []), _edges(100, 3), Edges([], [])]

        _assert_same(Edges.concatenate(parts), _fold(parts))

    def test_matches_fold_for_parts_without_attrs(self):
        parts = [_edges(0, 2), _edges(100, 3, with_attrs=False)]

        _assert_same(Edges.concatenate(parts), _fold(parts))

    def test_empty_input_matches_fold(self):
        result = Edges.concatenate([])

        assert len(result) == 0
        assert result.get_pairs().shape == (0, 2)
        _assert_same(result, _fold([]))

    def test_preserves_dtypes(self):
        result = Edges.concatenate([_edges(0, 2), _edges(100, 3)])

        assert result.node_ids1.dtype == basetypes.NODE_ID
        assert result.node_ids2.dtype == basetypes.NODE_ID
        assert result.affinities.dtype == _edges(0, 1).affinities.dtype
        assert result.areas.dtype == _edges(0, 1).areas.dtype

    def test_accepts_a_generator(self):
        """get_l2_agglomerations passes an itertools.chain, not a list."""
        parts = [_edges(0, 2), _edges(100, 3)]

        _assert_same(Edges.concatenate(chain(parts)), _fold(parts))

    def test_leaves_inputs_unmodified(self):
        parts = [_edges(0, 3), _edges(100, 2)]
        before = [p.node_ids1.copy() for p in parts]

        _ = Edges.concatenate(parts)

        for part, original in zip(parts, before):
            np.testing.assert_array_equal(part.node_ids1, original)

    def test_allocates_each_output_array_once(self):
        """The point of the change: n parts must not cost n concatenations."""
        parts = [_edges(i * 100, 2) for i in range(20)]
        calls = []
        original = np.concatenate

        def counting_concatenate(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        np.concatenate = counting_concatenate
        try:
            Edges.concatenate(parts)
            bulk_calls = len(calls)
            calls.clear()
            _fold(parts)
            fold_calls = len(calls)
        finally:
            np.concatenate = original

        assert bulk_calls == 4, f"expected one concatenate per array, got {bulk_calls}"
        assert fold_calls == 4 * len(parts)


def _svs(ids):
    return np.unique(np.array(ids, dtype=basetypes.NODE_ID))


class TestInSorted:
    def test_matches_np_isin(self):
        rng = np.random.default_rng(0)
        values = rng.integers(0, 200, size=500).astype(basetypes.NODE_ID)
        ref = _svs(rng.integers(0, 200, size=40))

        np.testing.assert_array_equal(in_sorted(values, ref), np.isin(values, ref))

    @pytest.mark.parametrize(
        "values,ref",
        [([], [1, 2]), ([1, 2], []), ([], []), ([5], [5]), ([5], [6])],
    )
    def test_edge_cases(self, values, ref):
        v, r = np.array(values, dtype=basetypes.NODE_ID), _svs(ref)
        np.testing.assert_array_equal(in_sorted(v, r), np.isin(v, r))

    def test_values_beyond_reference_range(self):
        """searchsorted returns len(ref) for values past the end; must not index out of bounds."""
        v = np.array([0, 999999], dtype=basetypes.NODE_ID)
        r = _svs([10, 20])

        np.testing.assert_array_equal(in_sorted(v, r), np.array([False, False]))


class TestFilterTouching:
    """Edges.filter_touching runs per chunk before edges are accumulated.

    The property that makes that sound: it must keep a superset of what the two
    consumers keep, so filtering early cannot change their output.
    """

    def _edges(self):
        # endpoints chosen to cover: both in, only first in, only second in, neither in
        return Edges(
            np.array([10, 20, 99, 98], dtype=basetypes.NODE_ID),
            np.array([11, 97, 30, 96], dtype=basetypes.NODE_ID),
            affinities=np.array([1, 2, 3, 4], dtype=basetypes.EDGE_AFFINITY),
            areas=np.array([5, 6, 7, 8], dtype=basetypes.EDGE_AREA),
        )

    def test_keeps_edges_touching_the_set(self):
        kept = self._edges().filter_touching(_svs([10, 11, 20, 30]))

        assert kept.node_ids1.tolist() == [10, 20, 99]
        assert kept.node_ids2.tolist() == [11, 97, 30]

    def test_carries_affinities_and_areas(self):
        kept = self._edges().filter_touching(_svs([10, 11, 20, 30]))

        assert kept.affinities.tolist() == [1, 2, 3]
        assert kept.areas.tolist() == [5, 6, 7]

    def test_is_superset_of_categorize_predicate(self):
        """categorize_edges_v2 only keeps edges whose node_ids1 is in the set."""
        e, svs = self._edges(), _svs([10, 11, 20, 30])
        kept = set(map(tuple, e.filter_touching(svs).get_pairs().tolist()))

        needed = {
            tuple(p) for p in e.get_pairs().tolist() if in_sorted(np.array([p[0]], dtype=basetypes.NODE_ID), svs)[0]
        }
        assert needed <= kept

    def test_is_superset_of_edges_only_predicate(self):
        """The edges_only path keeps edges with BOTH endpoints in the set."""
        e, svs = self._edges(), _svs([10, 11, 20, 30])
        kept = set(map(tuple, e.filter_touching(svs).get_pairs().tolist()))

        pairs = e.get_pairs()
        both = pairs[np.isin(pairs[:, 0], svs) & np.isin(pairs[:, 1], svs)]
        assert {tuple(p) for p in both.tolist()} <= kept

    def test_empty_set_drops_everything(self):
        kept = self._edges().filter_touching(_svs([]))

        assert len(kept) == 0
        assert kept.get_pairs().shape == (0, 2)

    def test_empty_edges(self):
        assert len(Edges([], []).filter_touching(_svs([1, 2]))) == 0

    def test_all_matching_is_identity(self):
        e = self._edges()
        kept = e.filter_touching(_svs([10, 11, 20, 97, 99, 30, 98, 96]))

        np.testing.assert_array_equal(kept.node_ids1, e.node_ids1)
        np.testing.assert_array_equal(kept.node_ids2, e.node_ids2)

    def test_filter_then_concatenate_equals_concatenate_then_filter(self):
        """Per-chunk filtering must equal filtering the fully accumulated set."""
        rng = np.random.default_rng(7)
        svs = _svs(rng.integers(0, 50, size=12))
        chunks = [
            Edges(
                rng.integers(0, 100, size=30).astype(basetypes.NODE_ID),
                rng.integers(0, 100, size=30).astype(basetypes.NODE_ID),
            )
            for _ in range(5)
        ]

        early = Edges.concatenate([c.filter_touching(svs) for c in chunks])
        late = Edges.concatenate(chunks).filter_touching(svs)

        np.testing.assert_array_equal(early.node_ids1, late.node_ids1)
        np.testing.assert_array_equal(early.node_ids2, late.node_ids2)
