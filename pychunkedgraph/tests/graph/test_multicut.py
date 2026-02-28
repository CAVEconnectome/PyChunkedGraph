import numpy as np
import pytest

from ...graph.edges import Edges
from ...graph import exceptions
from ...graph.cutting import run_multicut


class TestGraphMultiCut:
    @pytest.mark.timeout(30)
    def test_cut_multi_tree(self, gen_graph):
        """
        Multicut on a graph with multiple sources and sinks and parallel paths.
        Sources: [1, 2], Sinks: [5, 6]
        Graph:
           1━━3━━5
           ┃  ┃
           2━━4━━6
        The multicut should find edges to separate {1,2} from {5,6}.
        """
        node_ids1 = np.array([1, 2, 3, 4, 3, 1], dtype=np.uint64)
        node_ids2 = np.array([3, 4, 5, 6, 4, 2], dtype=np.uint64)
        affinities = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.9], dtype=np.float32)
        edges = Edges(node_ids1, node_ids2, affinities=affinities)
        source_ids = np.array([1, 2], dtype=np.uint64)
        sink_ids = np.array([5, 6], dtype=np.uint64)

        cut_edges = run_multicut(
            edges,
            source_ids,
            sink_ids,
            path_augment=False,
            disallow_isolating_cut=False,
        )
        assert cut_edges.shape[0] > 0

        # Verify the cut actually separates sources from sinks
        cut_set = set(map(tuple, cut_edges.tolist()))
        remaining = set()
        for i in range(len(node_ids1)):
            e = (int(node_ids1[i]), int(node_ids2[i]))
            if e not in cut_set and (e[1], e[0]) not in cut_set:
                remaining.add(e)

        # BFS from sources through remaining edges
        reachable = set(source_ids.tolist())
        changed = True
        while changed:
            changed = False
            for a, b in remaining:
                if a in reachable and b not in reachable:
                    reachable.add(b)
                    changed = True
                if b in reachable and a not in reachable:
                    reachable.add(a)
                    changed = True
        # Sinks should not be reachable from sources
        for s in sink_ids:
            assert int(s) not in reachable

    @pytest.mark.timeout(30)
    def test_path_augmented_multicut(self, sv_data):
        sv_edges, sv_sources, sv_sinks, sv_affinity, sv_area = sv_data
        edges = Edges(
            sv_edges[:, 0], sv_edges[:, 1], affinities=sv_affinity, areas=sv_area
        )
        cut_edges_aug = run_multicut(edges, sv_sources, sv_sinks, path_augment=True)
        assert cut_edges_aug.shape[0] == 350

        with pytest.raises(exceptions.PreconditionError):
            run_multicut(edges, sv_sources, sv_sinks, path_augment=False)
