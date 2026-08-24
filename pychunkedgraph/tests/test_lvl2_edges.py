"""Equivalence tests for ``_get_edges_for_lvl2_ids``.

The function was reworked to cut its peak memory (725MB -> 304MB on a synthetic object of
250k level 2 ids / 2M cross edges). Every change was supposed to be behaviour-preserving,
so what these tests pin is exactly that: a reference implementation of the previous
algorithm runs beside the current one over randomized inputs, and the two must agree
element for element. The reference is deliberately written the slow, obvious way -- it is
the specification, not an optimization.
"""

import numpy as np
import pytest

from ..graph.analysis.pathing import _get_edges_for_lvl2_ids


class FakeCg:
    """Just the two methods the function calls on a ChunkedGraph."""

    def __init__(self, cce_dict, parents=None):
        self._cce = cce_dict
        self._parents = parents or {}
        self.get_parents_calls = 0

    def get_atomic_cross_edges(self, l2_ids):
        # a fresh dict per call, as both real implementations return
        return {k: dict(v) for k, v in self._cce.items()}

    def get_parents(self, supervoxels):
        self.get_parents_calls += 1
        return np.array([self._parents[int(s)] for s in supervoxels], dtype=np.uint64)


def reference(cg, lvl2_ids, induced=False):
    """The previous implementation, verbatim in structure."""
    import fastremap

    if len(lvl2_ids) == 0:
        return np.empty((0, 2), dtype=np.uint64)
    cce_dict = cg.get_atomic_cross_edges(lvl2_ids)
    edge_array = []
    for l2_id in cce_dict:
        for level in cce_dict[l2_id]:
            edge_array.append(cce_dict[l2_id][level])
    if len(edge_array) == 0:
        return np.empty((0, 2), dtype=np.uint64)
    edge_array = np.concatenate(edge_array)
    known_sv, known_l2, unknown_sv = [], [], []
    for lvl2_id in cce_dict:
        for level in cce_dict[lvl2_id]:
            k = cce_dict[lvl2_id][level][:, 0]
            u = cce_dict[lvl2_id][level][:, 1]
            known_sv.append(k)
            known_l2.append(np.full(k.shape, lvl2_id))
            unknown_sv.append(u)
    known_supervoxel_array, unique_indices = np.unique(
        np.concatenate(known_sv), return_index=True
    )
    known_l2_array = (np.concatenate(known_l2))[unique_indices]
    unknown_supervoxel_array = np.unique(np.concatenate(unknown_sv))
    to_query = np.setdiff1d(unknown_supervoxel_array, known_supervoxel_array)
    if len(to_query) > 0:
        missing = cg.get_parents(to_query)
        known_supervoxel_array = np.concatenate((known_supervoxel_array, to_query))
        known_l2_array = np.concatenate((known_l2_array, missing))
    ev = edge_array.view()
    ev.shape = -1
    fastremap.remap_from_array_kv(ev, known_supervoxel_array, known_l2_array)
    edge_array = np.unique(np.sort(edge_array, axis=1), axis=0)
    if induced:
        edge_array = edge_array[
            np.isin(edge_array[:, 0], lvl2_ids) & np.isin(edge_array[:, 1], lvl2_ids)
        ]
    return edge_array


def build_case(seed, n_l2=60, layers=(2, 3), max_edges=4, outside_frac=0.3):
    """Random cross-edge dict plus the parent lookup for supervoxels outside it.

    Supervoxel ids are laid out so each belongs to exactly one level 2 node, which is what
    the real data guarantees and what the supervoxel -> parent mapping relies on.
    """
    rng = np.random.default_rng(seed)
    lvl2_ids = np.arange(1, n_l2 + 1, dtype=np.uint64) * 1000
    # supervoxel s belongs to lvl2 node (s // 1000) * 1000
    def svs_of(l2, count):
        return (np.uint64(l2) + rng.integers(1, 999, size=count)).astype(np.uint64)

    cce, parents = {}, {}
    outside_l2 = np.arange(n_l2 + 1, n_l2 + 21, dtype=np.uint64) * 1000
    for l2 in lvl2_ids:
        d = {}
        for layer in layers:
            n = int(rng.integers(0, max_edges + 1))
            if n == 0:
                d[layer] = np.empty((0, 2), dtype=np.uint64)
                continue
            col0 = svs_of(l2, n)
            partner_l2 = np.where(
                rng.random(n) < outside_frac,
                rng.choice(outside_l2, size=n),
                rng.choice(lvl2_ids, size=n),
            ).astype(np.uint64)
            col1 = np.array(
                [int(p) + int(rng.integers(1, 999)) for p in partner_l2], dtype=np.uint64
            )
            for sv, p in zip(col1.tolist(), partner_l2.tolist()):
                parents[int(sv)] = np.uint64(p)
            d[layer] = np.column_stack([col0, col1])
        cce[l2] = d
    # a supervoxel that appears in column 0 has a known parent; make sure the reference
    # and the implementation agree about those too
    for l2, d in cce.items():
        for layer, block in d.items():
            for sv in block[:, 0].tolist():
                parents.setdefault(int(sv), np.uint64(l2))
    return cce, parents, lvl2_ids


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("induced", [True, False])
def test_matches_previous_implementation(seed, induced):
    cce, parents, lvl2_ids = build_case(seed)
    got = _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, induced=induced)
    want = reference(FakeCg(cce, parents), lvl2_ids, induced=induced)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want), f"seed={seed} induced={induced}"


@pytest.mark.parametrize("seed", [0, 5])
def test_induced_edges_stay_inside_the_object(seed):
    cce, parents, lvl2_ids = build_case(seed)
    got = _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, induced=True)
    assert np.isin(got, lvl2_ids).all()


def test_result_is_deduplicated_and_sorted_within_each_edge():
    cce, parents, lvl2_ids = build_case(1)
    got = _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, induced=True)
    assert (got[:, 0] <= got[:, 1]).all(), "each edge should be sorted low, high"
    assert len(np.unique(got, axis=0)) == len(got), "no duplicate edges"


def test_no_lvl2_ids():
    out = _get_edges_for_lvl2_ids(FakeCg({}, {}), np.array([], dtype=np.uint64))
    assert out.shape == (0, 2) and out.dtype == np.uint64


def test_lvl2_ids_but_no_edges():
    lvl2_ids = np.array([1000, 2000], dtype=np.uint64)
    cce = {np.uint64(1000): {2: np.empty((0, 2), dtype=np.uint64)}, np.uint64(2000): {}}
    out = _get_edges_for_lvl2_ids(FakeCg(cce, {}), lvl2_ids, induced=True)
    assert out.shape == (0, 2) and out.dtype == np.uint64


def test_parents_are_fetched_only_for_unknown_supervoxels():
    """Supervoxels seen in column 0 must not be re-queried against the graph."""
    cce, parents, lvl2_ids = build_case(3)
    cg = FakeCg(cce, parents)
    _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
    assert cg.get_parents_calls == 1


def test_caller_lvl2_ids_not_mutated():
    """np.unique / sort inside must not reorder the array the caller passed in."""
    cce, parents, lvl2_ids = build_case(7)
    shuffled = lvl2_ids.copy()
    np.random.default_rng(0).shuffle(shuffled)
    before = shuffled.copy()
    _get_edges_for_lvl2_ids(FakeCg(cce, parents), shuffled, induced=True)
    assert np.array_equal(shuffled, before)


def test_duplicate_lvl2_ids_in_input():
    cce, parents, lvl2_ids = build_case(2)
    dup = np.concatenate([lvl2_ids, lvl2_ids[:10]])
    got = _get_edges_for_lvl2_ids(FakeCg(cce, parents), dup, induced=True)
    want = _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, induced=True)
    assert np.array_equal(got, want)
