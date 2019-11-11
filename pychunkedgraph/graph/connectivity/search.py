import random
from typing import List

import numpy as np
from graph_tool.search import bfs_search
from graph_tool.search import BFSVisitor
from graph_tool.search import StopSearch

from ..utils.basetypes import NODE_ID


class TargetVisitor(BFSVisitor):
    def __init__(self, target, reachable):
        self.target = target
        self.reachable = reachable

    def discover_vertex(self, u):
        if u == self.target:
            self.reachable[u] = 1
            raise StopSearch


def check_reachability(g, sv1s: np.ndarray, sv2s: np.ndarray, original_ids: np.ndarray) -> np.ndarray:
    """
    g: graph tool Graph instance with ids 0 to N-1 where N = vertex count
    original_ids: sorted ChunkedGraph supervoxel ids
        (to identify corresponding ids in graph tool)
    for each pair (sv1, sv2) check if a path exists (BFS)
    """
    # mapping from original ids to graph tool ids
    original_ids_d = {
        sv_id: index for sv_id, index in zip(original_ids, range(len(original_ids)))
    }
    reachable = g.new_vertex_property("int", val=0)

    def _check_reachability(source, target):
        bfs_search(g, source, TargetVisitor(target, reachable))
        return reachable[target]

    return np.array(
        [
            _check_reachability(original_ids_d[source], original_ids_d[target])
            for source, target in zip(sv1s, sv2s)
        ],
        dtype=bool,
    )

