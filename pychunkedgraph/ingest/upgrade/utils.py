from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Hierarchy


def exists_as_parent(cg: ChunkedGraph, parent, nodes) -> bool:
    """
    Check if a given l2 parent is in the history of given nodes.
    """
    response = cg.client.read_nodes(node_ids=nodes, properties=Hierarchy.Parent)
    parents = set()
    for cells in response.values():
        parents.update([cell.value for cell in cells])
    return parent in parents
