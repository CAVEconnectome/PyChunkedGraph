from time import time
from ..graph import ChunkedGraph

cg = ChunkedGraph(graph_id="minnie3_v0")


def get_subgraph(node_id):
    start = time()
    result = cg.get_subgraph(node_id, nodes_only=True)
    print("cg.get_subgraph", time() - start)
    return result
