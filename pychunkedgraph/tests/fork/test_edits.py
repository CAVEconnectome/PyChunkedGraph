"""A fork's edits change its own roots alone, minting ids above its source's."""

from datetime import UTC, datetime


def _roots(graph, ids) -> int:
    """How many roots `ids` climb to in `graph`."""
    return len(set(graph.get_roots(ids)))


def test_a_merge_on_a_fork_joins_its_roots_and_not_the_source_s(apart, fork_of):
    cg, ids = apart
    fork = fork_of(cg, "test_fork", datetime.now(UTC))
    fork.add_edges("test_user", list(ids), affinities=[0.5])
    assert (_roots(fork, ids), _roots(cg, ids)) == (1, 2)


def test_a_split_on_a_fork_parts_its_roots_and_not_the_source_s(joined, fork_of):
    cg, ids = joined
    fork = fork_of(cg, "test_fork", datetime.now(UTC))
    fork.remove_edges("test_user", source_ids=ids[0], sink_ids=ids[1], mincut=False)
    assert (_roots(fork, ids), _roots(cg, ids)) == (2, 1)


def test_a_fork_edit_mints_roots_above_every_root_its_source_minted(joined, fork_of):
    cg, ids = joined
    fork = fork_of(cg, "test_fork", datetime.now(UTC))
    # The source's split mints roots from its counters after the fork point.
    cg.remove_edges("test_user", source_ids=ids[0], sink_ids=ids[1], mincut=False)
    fork.remove_edges("test_user", source_ids=ids[0], sink_ids=ids[1], mincut=False)
    assert fork.get_roots(ids).min() > cg.get_roots(ids).max()
