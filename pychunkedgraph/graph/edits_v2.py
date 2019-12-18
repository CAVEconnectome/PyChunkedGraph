import datetime
import numpy as np
from collections import defaultdict
from typing import Sequence


from .utils import basetypes
from .utils import flatgraph
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists


def _process_atomic_edge(cg, atomic_edges):
    """
    Determine if the edge is within the chunk.
    If not, consider it as a cross edge between two L2 IDs in different chunks.
    """
    lvl2_edges = []
    edge_layers = cg.get_cross_chunk_edges_layer(atomic_edges)
    edge_layer_m = edge_layers > 1

    cross_edge_dict = {}
    for atomic_edge in atomic_edges[~edge_layer_m]:
        lvl2_edges.append(
            [cg.get_parent(atomic_edge[0]), cg.get_parent(atomic_edge[1])]
        )

    for atomic_edge, layer in zip(
        atomic_edges[edge_layer_m], edge_layers[edge_layer_m]
    ):
        parent_id_0 = cg.get_parent(atomic_edge[0])
        parent_id_1 = cg.get_parent(atomic_edge[1])

        cross_edge_dict[parent_id_0] = {layer: atomic_edge}
        cross_edge_dict[parent_id_1] = {layer: atomic_edge[::-1]}

        lvl2_edges.append([parent_id_0, parent_id_0])
        lvl2_edges.append([parent_id_1, parent_id_1])
    return lvl2_edges, cross_edge_dict


def add_edge_v2(
    cg,
    *,
    edge: np.ndarray,
    operation_id: np.uint64 = None,
    source_coords: Sequence[np.uint64] = None,
    sink_coords: Sequence[np.uint64] = None,
    timestamp: datetime.datetime = None,
):
    """
    # if there is no path between sv1 and sv2 (edge)
    # in the subgraph, add "fake" edges, these are stored in a row per chunk

    get level 2 ids for both roots
    create a new level 2 id for ids that have a linking edge
    merge the cross edges pf these ids into new id
    """
    l2id_agg_d = cg.get_subgraph(
        agglomeration_ids=np.unique(cg.get_roots(edge.ravel())),
        bbox=get_bounding_box(source_coords, sink_coords),
        bbox_is_coordinate=True,
        timestamp=timestamp,
    )
    l2ids = np.fromiter(l2id_agg_d.keys(), dtype=basetypes.NODE_ID)
    chunk_ids = cg.get_chunk_ids_from_node_ids(l2ids)

    chunk_l2ids_d = defaultdict(list)
    for idx, l2id in enumerate(l2ids):
        chunk_l2ids_d[chunk_ids[idx]].append(l2id)

    # There needs to be atleast one inactive edge between
    # supervoxels in the sub-graph (within bounding box)
    # for merging two root ids without a fake edge

    # add_fake_edge = False
    # for aggs in chunk_l2ids_d.values():
    #     if edge_exists(aggs):
    #         add_fake_edge = True
    #         break

    parent_1, parent_2 = cg.get_parents(edge)
    chunk_id1, chunk_id2 = cg.get_chunk_ids_from_node_ids(edge)

    # TODO add read cross chunk edges method to client
    # try merge with existing functions
