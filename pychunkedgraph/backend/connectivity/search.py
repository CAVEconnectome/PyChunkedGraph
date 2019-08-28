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
            self.reachable[u] = True
            raise StopSearch


def check_reachability(g, sv1s, sv2s, original_ids) -> np.ndarray:
    """
    for each pair (sv1, sv2) check if a path exists (BFS)
    """
    # mapping from original ids to graph tool ids
    original_ids_d = {
        sv_id: index for sv_id, index in zip(original_ids, range(len(original_ids)))
    }
    reachable = g.new_vertex_property("bool", val=False)
    print(g.vertex_properties)

    def _check_reachability(source, target):
        bfs_search(g, source, TargetVisitor(target, reachable))
        print(reachable[target])
        return reachable[target]

    return np.array(
        [
            _check_reachability(original_ids_d[source], original_ids_d[target])
            for source, target in zip(sv1s, sv2s)
        ]
    )

