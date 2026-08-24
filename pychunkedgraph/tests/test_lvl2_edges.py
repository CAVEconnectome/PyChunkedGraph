"""Equivalence tests for ``_get_edges_for_lvl2_ids``.

The function was reworked to cut its peak memory -- measured in the api6 read pod on
aibs_v1dd root 864691132764660103, peak rss 1868-1878MB -> 991-1031MB, and slightly
faster with it. Every change was meant to be behaviour-preserving, so that is what these
tests pin: a reference implementation of the previous algorithm runs beside the current
one over randomized inputs, and the two must agree element for element. The reference is
deliberately written the slow, obvious way -- it is the specification, not an
optimization.

The read is now sliced, so the tests also cover the slicing itself: that the result does
not depend on the slice size, that every id is read exactly once (one Bigtable pass, not
two), and that a partner living in a different slice is still resolved.
"""

import importlib

import numpy as np
import pytest

from ..graph.analysis.pathing import _get_edges_for_lvl2_ids


class FakeCg:
    """Just the two methods the function calls on a ChunkedGraph."""

    def __init__(self, cce_dict, parents=None):
        self._cce = cce_dict
        self._parents = parents or {}
        self.get_parents_calls = 0
        self.read_calls = 0
        self.ids_read = []

    def get_atomic_cross_edges(self, l2_ids):
        # A fresh dict per call holding only the ids asked for, as the real one does.
        # Honouring l2_ids matters: the implementation calls this once per slice, and a
        # fake that ignored the argument would hand every slice the whole object, so no
        # slicing bug could ever fail a test.
        wanted = {int(i) for i in l2_ids}
        self.read_calls += 1
        self.ids_read.extend(sorted(wanted))
        return {k: dict(v) for k, v in self._cce.items() if int(k) in wanted}

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


@pytest.mark.parametrize("batch_size", [1, 2, 7, 59, 10_000])
@pytest.mark.parametrize("induced", [True, False])
def test_result_is_independent_of_batch_size(batch_size, induced):
    """Slicing the read must change the peak memory and nothing else."""
    cce, parents, lvl2_ids = build_case(4)
    want = reference(FakeCg(cce, parents), lvl2_ids, induced=induced)
    got = _get_edges_for_lvl2_ids(
        FakeCg(cce, parents), lvl2_ids, induced=induced, batch_size=batch_size
    )
    assert np.array_equal(got, want), f"batch_size={batch_size} induced={induced}"


def test_read_is_sliced_and_covers_every_id_once():
    """One read per slice, every id read exactly once -- a single pass over Bigtable.

    Asserted directly because a slicing bug that skipped or repeated ids would still
    usually produce plausible-looking output.
    """
    cce, parents, lvl2_ids = build_case(6)
    cg = FakeCg(cce, parents)
    _get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True, batch_size=7)
    assert cg.read_calls == int(np.ceil(len(lvl2_ids) / 7))
    ids, counts = np.unique(np.array(cg.ids_read), return_counts=True)
    assert np.array_equal(ids, np.unique(lvl2_ids)), "every id must be read"
    assert set(counts.tolist()) == {1}, "and read exactly once -- one pass, not two"


def test_partner_in_a_different_slice_is_still_resolved():
    """The mapping must be complete before anything is remapped; with batch_size=1 every
    partner necessarily belongs to a different slice."""
    cce, parents, lvl2_ids = build_case(8)
    want = reference(FakeCg(cce, parents), lvl2_ids, induced=True)
    got = _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, induced=True, batch_size=1)
    assert np.array_equal(got, want)
    assert len(got) > 0, "fixture should produce cross-slice edges"


def test_subset_query_still_looks_up_outside_parents():
    """A bounds-style query: ask for half the object, so partners outside it have no
    mapping entry and have to come from get_parents."""
    cce, parents, lvl2_ids = build_case(9)
    subset = lvl2_ids[: len(lvl2_ids) // 2]
    cg = FakeCg(cce, parents)
    got = _get_edges_for_lvl2_ids(cg, subset, induced=True, batch_size=5)
    want = reference(FakeCg(cce, parents), subset, induced=True)
    assert np.array_equal(got, want)
    assert cg.get_parents_calls == 1, "partners outside the subset must be looked up"
    assert np.isin(got, subset).all()


def _reload_pathing():
    """Re-import the module so its module-level env parsing runs again."""
    from pychunkedgraph.graph.analysis import pathing

    return importlib.reload(pathing)


@pytest.fixture
def pathing_module():
    """Reload around the test so an env override cannot leak into the rest of the suite."""
    yield _reload_pathing()
    _reload_pathing()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2500", 2500),
        ("1", 1),
        (None, 10_000),          # unset -> default
        ("not-a-number", 10_000),  # unparseable -> default, not a crash at import
        ("0", 10_000),           # would yield no slices at all -> refused
        ("-5", 10_000),          # same
    ],
)
def test_batch_size_env_var(monkeypatch, pathing_module, value, expected):
    if value is None:
        monkeypatch.delenv("PCG_CROSS_EDGE_READ_BATCH", raising=False)
    else:
        monkeypatch.setenv("PCG_CROSS_EDGE_READ_BATCH", value)
    assert _reload_pathing().CROSS_EDGE_READ_BATCH == expected


def test_env_batch_size_is_actually_used(monkeypatch, pathing_module):
    """The constant has to reach the read loop, not just sit in the module."""
    monkeypatch.setenv("PCG_CROSS_EDGE_READ_BATCH", "9")
    mod = _reload_pathing()
    cce, parents, lvl2_ids = build_case(6)
    cg = FakeCg(cce, parents)
    got = mod._get_edges_for_lvl2_ids(cg, lvl2_ids, induced=True)
    assert cg.read_calls == int(np.ceil(len(lvl2_ids) / 9))
    assert np.array_equal(got, reference(FakeCg(cce, parents), lvl2_ids, induced=True))


def test_explicit_non_positive_batch_size_raises():
    """Silently returning zero edges would be far worse than failing."""
    cce, parents, lvl2_ids = build_case(0)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            _get_edges_for_lvl2_ids(FakeCg(cce, parents), lvl2_ids, batch_size=bad)
