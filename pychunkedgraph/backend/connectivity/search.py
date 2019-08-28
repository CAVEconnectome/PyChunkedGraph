from typing import List

from graph_tool.search import bfs_search
from graph_tool.search import BFSVisitor
from graph_tool.search import StopSearch

from ..utils.basetypes import NODE_ID


class TargetVisitor(BFSVisitor):
    def __init__(self, target):
        self.target = target

    def discover_vertex(self, u: NODE_ID):
        if u == self.target:
            raise StopSearch


def check_reachability(g, sv1s, sv2s) -> List[bool]:
    """
    for each pair (sv1, sv2) check if a path exists (BFS)
    """

    def _check_reachability(source, target):
        try:
            bfs_search(g, source, TargetVisitor(target))
        except StopSearch:
            return True
        return False

    return [_check_reachability(source, target) for source, target in zip(sv1s, sv2s)]

