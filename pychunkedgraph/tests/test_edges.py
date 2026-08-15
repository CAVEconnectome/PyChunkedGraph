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

from ..graph.edges import Edges
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
