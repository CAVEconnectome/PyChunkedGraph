# pylint: disable=invalid-name, missing-docstring, c-extension-no-member
import fastremap
import numpy as np
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Connectivity
from pychunkedgraph.graph.utils import serializers
from pychunkedgraph.graph.edges.utils import concatenate_cross_edge_dicts


def get_edit_timestamps(cg: ChunkedGraph, children: np.ndarray) -> set:
    """
    Collect timestamps of edits from children, since we use the same timestamp
    for all IDs involved in an edit, we can use the timestamps of
    when cross edges of children were updated.
    """
    response = cg.client.read_nodes(node_ids=children)
    result = set()
    for v in response.values():
        for layer in range(2, cg.meta.layer_count):
            col = Connectivity.CrossChunkEdge[layer]
            if col not in v:
                continue
            for cell in v[col]:
                result.add(cell.timestamp)
    return result


def update_cross_edges(cg: ChunkedGraph, layer, node, children, timestamps) -> list:
    """
    Helper function to update a single ID.
    Returns a list of mutations with given timestamps.
    """
    rows = []
    for ts in sorted(timestamps):
        cx_edges_d = cg.get_cross_chunk_edges(children, time_stamp=ts, raw_only=True)
        cx_edges_d = concatenate_cross_edge_dicts(cx_edges_d.values())
        edges = np.concatenate(list(cx_edges_d.values()))
        assert node == np.unique(cg.get_parents(edges[:, 0], time_stamp=ts))

        val_dict = {}
        nodes = edges[:, 1]
        # parents = cg.get_parents(nodes, time_stamp=ts)
        parents = cg.get_roots(nodes, time_stamp=ts, stop_layer=layer, ceil=False)
        edge_parents_d = dict(zip(nodes, parents))
        for layer, layer_edges in cx_edges_d.items():
            layer_edges = fastremap.remap(
                layer_edges, edge_parents_d, preserve_missing_labels=True
            )
            layer_edges[:, 0] = node
            layer_edges = np.unique(layer_edges, axis=0)
            col = Connectivity.CrossChunkEdge[layer]
            val_dict[col] = layer_edges
        row_id = serializers.serialize_uint64(node)
        rows.append(cg.client.mutate_row(row_id, val_dict, time_stamp=ts))
    return rows


def update_chunk(cg: ChunkedGraph, chunk_coords: list[int], layer: int):
    """
    Iterate over all layer IDs in a chunk and update their cross chunk edges.
    """
    x, y, z = chunk_coords
    rr = cg.range_read_chunk(cg.get_chunk_id(layer=layer, x=x, y=y, z=z))
    nodes = list(rr.keys())
    children_d = cg.get_children(nodes)

    rows = []
    for node in nodes:
        timestamps = get_edit_timestamps(cg, children_d[node])
        _rows = update_cross_edges(cg, layer, node, children_d[node], timestamps)
        rows.extend(_rows)
    cg.client.write(rows)
