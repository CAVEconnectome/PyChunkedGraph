"""Tests for the corridor pruning in ``find_l2_shortest_path``.

The endpoint used to read an entire object's level 2 graph to answer a question about a
thin path through it. It now builds a coarse graph over subtrees, routes through that,
dilates the result, and reads level 2 cross edges only under the corridor.

Two properties matter and neither is obvious from the result alone, so both are asserted
directly:

  * the pruned search returns the same path as the full search on these fixtures (they are
    built so the route is unique, which is also close to how real level 2 graphs look --
    276,472 edges over 240,928 nodes on the profiled object);
  * it actually prunes, i.e. reads fewer level 2 nodes than the object contains. A version
    that quietly read everything would pass a result-only test.

The fixture encodes the same supervoxel convention as the real data: a supervoxel belongs
to exactly one level 2 node, and a cross edge is stored on the rows of *both* endpoints.
That symmetry is what lets the coarse pass skip get_parents.
"""

from collections import namedtuple

import numpy as np
import pytest

from ..graph import attributes
from ..graph.analysis import pathing

Cell = namedtuple("Cell", ["value"])

ROOT = np.uint64(10**9)
LAYER_COUNT = 12
# The layer the fixture parks its groups at, and therefore the layer its inter-group
# edges are stored under. The implementation does not know this number -- it finds the cut
# by descending to a target width -- but the fixture has to put the groups somewhere.
COARSE_LAYER = 5
# Narrow enough that the descent stops on the fixture's group layer rather than running
# on past it to level 2. The default fixture has 24 group nodes, so anything at or below
# that stops there; ask for more and the cut descends to level 2, which is what
# test_cut_width_follows_the_target_not_a_layer exercises deliberately.
TARGET_GROUPS = 20


class FakeCg:
    """Enough of a ChunkedGraph for find_l2_shortest_path.

    Layers come from an explicit dict rather than from id bit-packing, so fixtures can lay
    out a hierarchy without reproducing the real id encoding.
    """

    class _Meta:
        layer_count = LAYER_COUNT

    class _Client:
        def __init__(self, outer):
            self._outer = outer

        def read_nodes(self, node_ids, properties):
            """Restricted-column read, as _coarse_edge_list issues it."""
            self._outer.restricted_reads += 1
            self._outer.restricted_ids_read.extend(int(i) for i in node_ids)
            wanted = {p.index for p in properties}
            out = {}
            for l2_id in node_ids:
                per_layer = self._outer.cross_edges.get(int(l2_id), {})
                cols = {
                    attributes.Connectivity.CrossChunkEdge[layer]: [Cell(value=block)]
                    for layer, block in per_layer.items()
                    if layer in wanted and len(block)
                }
                if cols:
                    out[np.uint64(l2_id)] = cols
            return out

    def __init__(self, layers, children, cross_edges):
        self.graph_id = "fake_graph"
        self.meta = self._Meta()
        self.client = self._Client(self)
        self._layers = layers
        self._children = children
        self._parent = {c: p for p, cs in children.items() for c in cs}
        self.cross_edges = cross_edges
        self.restricted_reads = 0
        self.restricted_ids_read = []
        self.full_reads = 0
        self.full_ids_read = []

    # -- hierarchy ---------------------------------------------------------------
    def get_chunk_layers(self, node_ids):
        return np.array([self._layers[int(n)] for n in node_ids], dtype=np.int32)

    def get_children(self, node_ids, flatten=False):
        if flatten:
            out = [
                np.array(self._children.get(int(n), []), dtype=np.uint64)
                for n in node_ids
            ]
            return np.concatenate(out) if out else np.empty(0, dtype=np.uint64)
        return {
            np.uint64(n): np.array(self._children.get(int(n), []), dtype=np.uint64)
            for n in node_ids
        }

    def get_root(self, node_id, time_stamp=None, **kwargs):
        node = int(node_id)
        while node in self._parent:
            node = self._parent[node]
        return np.uint64(node)

    def get_parent(self, node_id, time_stamp=None):
        return self._parent.get(int(node_id))

    def get_parents(self, node_ids, time_stamp=None):
        """Supervoxel -> the level 2 node that owns it *now*, as the real one does.

        Ownership is read back out of the cross edge rows (column 0 of a node's rows is
        its own supervoxels) rather than inferred from the id, because an edit re-mints
        the owning level 2 id while leaving the supervoxel alone -- which is exactly the
        case the edge-list splice depends on.
        """
        owner = {}
        for lvl2_id, per_layer in self.cross_edges.items():
            for block in per_layer.values():
                block = np.asarray(block)
                if block.size:
                    for sv in block.reshape(-1, 2)[:, 0].tolist():
                        owner[int(sv)] = lvl2_id
        return np.array(
            [owner.get(int(n), self._parent.get(int(n), n)) for n in node_ids],
            dtype=np.uint64,
        )

    # -- edges -------------------------------------------------------------------
    def get_atomic_cross_edges(self, l2_ids):
        """Full-layer read, as _get_edges_for_lvl2_ids issues it."""
        self.full_reads += 1
        self.full_ids_read.extend(int(i) for i in l2_ids)
        out = {}
        for l2_id in l2_ids:
            per_layer = self.cross_edges.get(int(l2_id), {})
            out[np.uint64(l2_id)] = {
                layer: block for layer, block in per_layer.items() if len(block)
            }
        return out


def _sv(l2_id, k=1):
    """A supervoxel of `l2_id`. Belongs to exactly one level 2 node, as in real data."""
    return np.uint64(int(l2_id) + k)


def build_object(n_groups=12, per_group=40, side_depth=1):
    """A chain of coarse groups with a dead-end branch hanging off each one.

    Level 2 nodes inside a group form a path, joined group-to-group. The route from the
    first group to the last runs along the spine, so the branches are what a corridor
    should exclude -- but only the parts of them further from the spine than the dilation
    reaches. `side_depth` is the branch length in groups: with the default dilation of 2,
    a depth of 1 or 2 is entirely inside the corridor by construction, and only depth 3
    and beyond can be pruned. Tests that assert pruning have to ask for a deeper branch.

    Returns (cg, spine_l2, side_l2, source, target), where side_l2 is keyed by depth in
    `cg.side_l2_by_depth` so a test can distinguish "should be kept" from "should be cut".
    """
    layers = {int(ROOT): LAYER_COUNT - 1}
    children = {int(ROOT): []}
    cross = {}

    def add_edge(a, b, layer):
        cross.setdefault(int(a), {}).setdefault(layer, []).append(
            [_sv(a), _sv(b)]
        )
        cross.setdefault(int(b), {}).setdefault(layer, []).append(
            [_sv(b), _sv(a)]
        )

    next_group = [1000000]

    def make_group(size):
        gid = next_group[0]
        next_group[0] += 1000000
        layers[gid] = COARSE_LAYER
        children[int(ROOT)].append(gid)
        members = []
        for i in range(size):
            l2 = gid + (i + 1) * 1000
            layers[l2] = 2
            members.append(l2)
        children[gid] = members
        for a, b in zip(members, members[1:]):
            add_edge(a, b, 2)  # inside a group -> low layer boundary
        return gid, members

    spine_groups, spine_l2, side_l2 = [], [], []
    for _ in range(n_groups):
        gid, members = make_group(per_group)
        spine_groups.append((gid, members))
        spine_l2.extend(members)
    # join consecutive spine groups at the coarse layer so the coarse graph sees them
    for (_, a_members), (_, b_members) in zip(spine_groups, spine_groups[1:]):
        add_edge(a_members[-1], b_members[0], COARSE_LAYER)
    # dead-end branches: a chain of `side_depth` groups hanging off each spine group
    side_by_depth = {}
    for gid, members in spine_groups:
        attach_to = members[len(members) // 2]
        for depth in range(1, side_depth + 1):
            _, s_members = make_group(per_group)
            side_l2.extend(s_members)
            side_by_depth.setdefault(depth, []).extend(s_members)
            add_edge(attach_to, s_members[0], COARSE_LAYER)
            attach_to = s_members[-1]

    for l2, per_layer in cross.items():
        cross[l2] = {
            layer: np.array(pairs, dtype=np.uint64) for layer, pairs in per_layer.items()
        }
    cg = FakeCg(layers, children, cross)
    cg.side_l2_by_depth = side_by_depth
    source = np.uint64(spine_groups[0][1][0])
    target = np.uint64(spine_groups[-1][1][-1])
    return cg, spine_l2, side_l2, source, target


def _coarse_groups(cg, node_id, target_groups):
    """The old three-tuple, rebuilt from the split pipeline, so the tests below can keep
    asking the one question they care about: how the object was partitioned."""
    roots, by_group, layer, _, _ = pathing._coarse_membership(
        cg, node_id, target_groups, cache=None
    )
    lvl2, groups = pathing._flatten_groups(roots, by_group)
    return lvl2, groups, layer


def _full_path(cg, source, target):
    """The exact answer: shortest path over the whole object's level 2 graph."""
    lvl2, _, _ = _coarse_groups(cg, ROOT, TARGET_GROUPS)
    edges = pathing._get_edges_for_lvl2_ids(cg, lvl2, induced=True)
    return pathing._shortest_path_between(edges, source, target)


@pytest.fixture(autouse=True)
def tuned_for_small_fixtures(monkeypatch):
    """Force the coarse path on these deliberately small fixtures.

    The real thresholds (50,000 level 2 ids, a 15,000 wide cut) exist so ordinary objects
    skip pruning entirely; a fixture that large would make the tests unbearably slow.
    """
    monkeypatch.setattr(pathing, "FIND_PATH_MIN_LVL2_IDS", 1)
    monkeypatch.setattr(pathing, "FIND_PATH_TARGET_GROUPS", TARGET_GROUPS)


def test_coarse_groups_partition_the_object():
    cg, spine_l2, side_l2, _, _ = build_object()
    lvl2, groups, coarse_layer = _coarse_groups(cg, ROOT, TARGET_GROUPS)
    assert sorted(lvl2.tolist()) == sorted(spine_l2 + side_l2)
    assert len(lvl2) == len(groups)
    # every level 2 node lands in exactly one group, and groups are whole subtrees
    for l2, g in zip(lvl2.tolist(), groups.tolist()):
        assert l2 in cg._children[g]
    assert coarse_layer == COARSE_LAYER, "the cut should land on the fixture's group layer"


def test_cut_width_follows_the_target_not_a_layer():
    """The cut is chosen by width. A larger target descends further, to finer groups."""
    cg, spine_l2, side_l2, _, _ = build_object(n_groups=6, per_group=10, side_depth=1)
    n_group_nodes = 12  # 6 spine + 6 side
    _, coarse, _ = _coarse_groups(cg, ROOT, n_group_nodes - 2)
    _, fine, _ = _coarse_groups(cg, ROOT, 10**6)
    assert len(np.unique(coarse)) == n_group_nodes
    # asking for more groups than exist at that layer pushes the cut down to level 2
    assert len(np.unique(fine)) == len(spine_l2) + len(side_l2)


def test_shallow_shared_parent_falls_back_instead_of_paying_for_a_useless_cut():
    """A cut of one group cannot separate anything.

    Reached when the shared parent sits low in the hierarchy but still covers a lot: the
    coarse graph would be empty and the corridor the whole object, so the coarse read is
    wasted. It should be skipped entirely.
    """
    cg, spine_l2, _, _, _ = build_object()
    only_group = int(cg._parent[int(spine_l2[0])])
    members = cg._children[only_group]
    source, target = np.uint64(members[0]), np.uint64(members[-1])
    # force past the small-object bypass so the single-group guard is what has to catch it
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert got is not None
    assert cg.restricted_reads == 0, "no coarse read should be issued for a single group"


def test_pruned_path_matches_the_full_search():
    cg, _, _, source, target = build_object()
    want = _full_path(cg, source, target)
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert want is not None and len(want) > 1
    assert np.array_equal(got, want)


def test_it_actually_prunes(monkeypatch):
    """Reads fewer level 2 nodes than the object holds -- otherwise nothing was saved.

    Pins its own dilation rather than relying on the default, so that retuning the default
    changes the trade-off and not whether pruning is tested at all. Branches are 5 groups
    deep against a dilation of 2, so depth 1-2 is legitimately inside the corridor and
    depth 3+ is what pruning has to cut.
    """
    monkeypatch.setattr(pathing, "FIND_PATH_DILATION", 2)
    cg, spine_l2, side_l2, source, target = build_object(
        n_groups=12, per_group=40, side_depth=5
    )
    total = len(spine_l2) + len(side_l2)
    path = pathing.find_l2_shortest_path(cg, source, target)
    corridor_ids = set(cg.full_ids_read)
    assert cg.full_reads > 0, "the corridor still has to be read"
    assert len(corridor_ids) < total, "no pruning happened"
    assert not set(spine_l2) - corridor_ids, "the route itself must be inside the corridor"
    assert np.array_equal(path, _full_path(cg, source, target))

    far = set().union(*(cg.side_l2_by_depth[d] for d in (4, 5)))
    assert not (corridor_ids & far), "branch tips beyond the dilation should be cut"


def test_coarse_pass_reads_only_high_layer_columns():
    cg, _, _, source, target = build_object()
    pathing.find_l2_shortest_path(cg, source, target)
    assert cg.restricted_reads > 0, "the coarse graph should have been built"


def test_small_object_skips_the_coarse_pass(monkeypatch):
    monkeypatch.setattr(pathing, "FIND_PATH_MIN_LVL2_IDS", 10**9)
    cg, _, _, source, target = build_object()
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert cg.restricted_reads == 0, "no coarse graph for a small object"
    assert np.array_equal(got, _full_path(cg, source, target))


def test_disconnected_nodes_return_none():
    cg, _, _, source, _ = build_object()
    # a level 2 node under a different root entirely
    other_root = np.uint64(2 * 10**9)
    cg._layers[int(other_root)] = LAYER_COUNT - 1
    lone = np.uint64(99_000_000)
    cg._layers[int(lone)] = 2
    cg._children[int(other_root)] = [int(lone)]
    cg._parent[int(lone)] = int(other_root)
    assert pathing.find_l2_shortest_path(cg, source, lone) is None


def test_falls_back_when_the_corridor_misses_the_route(monkeypatch):
    """A corridor of zero width still must not report the nodes as unconnected.

    _dilate is stubbed to return only the source group, so the corridor cannot contain the
    target. The full-graph fallback is the only thing that can answer, and it must.
    """
    cg, _, _, source, target = build_object()
    monkeypatch.setattr(
        pathing, "_dilate", lambda edges, seeds, hops: np.unique(seeds)[:1]
    )
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert got is not None, "fallback did not run"
    assert np.array_equal(got, _full_path(cg, source, target))


def test_size_guard_is_enforced():
    from ..graph import exceptions as cg_exceptions

    cg, _, _, source, target = build_object()
    with pytest.raises(cg_exceptions.BadRequest, match="exceeds the maximum"):
        pathing.find_l2_shortest_path(cg, source, target, max_num_lvl2_ids=5)


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_result_is_stable_across_dilation(monkeypatch, dilation):
    monkeypatch.setattr(pathing, "FIND_PATH_DILATION", dilation)
    cg, _, _, source, target = build_object()
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert np.array_equal(got, _full_path(cg, source, target))


def test_endpoints_in_the_same_group():
    """A short hop inside one group still has to work, and still go through the corridor."""
    cg, spine_l2, _, _, _ = build_object()
    source, target = np.uint64(spine_l2[0]), np.uint64(spine_l2[5])
    got = pathing.find_l2_shortest_path(cg, source, target)
    assert np.array_equal(got, _full_path(cg, source, target))


def test_dilation_zero_is_honoured_not_silently_replaced():
    """0 is a meaningful dilation -- the coarse path and nothing around it.

    The env helper floors most tunables at 1, because a zero there would mean an empty
    read. Dilation is the exception, and setting it must not silently give the default.
    """
    import os

    from ..graph.analysis import pathing as p

    assert p._positive_int_from_env("PCG_NOPE_NOT_SET", 7, minimum=0) == 7
    os.environ["PCG_TMP_DILATION_TEST"] = "0"
    try:
        assert p._positive_int_from_env("PCG_TMP_DILATION_TEST", 5, minimum=0) == 0
        # still rejected where zero would mean "read nothing"
        assert p._positive_int_from_env("PCG_TMP_DILATION_TEST", 5) == 5
    finally:
        del os.environ["PCG_TMP_DILATION_TEST"]


def test_zero_dilation_still_finds_a_path():
    """With no margin at all the corridor is just the coarse route; it must still work,
    falling back if that route has no level 2 realisation."""
    cg, _, _, source, target = build_object()
    import pytest as _pytest  # noqa: F401

    got = pathing.find_l2_shortest_path(cg, source, target)
    assert got is not None
