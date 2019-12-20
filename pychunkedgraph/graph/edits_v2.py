import datetime
import numpy as np
from collections import defaultdict
from typing import Dict
from typing import Tuple
from typing import Iterable
from typing import Sequence

from .utils import basetypes
from .utils import flatgraph
from .utils.generic import get_bounding_box
from .connectivity.nodes import edge_exists


def _analyze_atomic_edge(cg, edge) -> Tuple[Iterable, Dict]:
    """
    Determine if the edge is within the chunk.
    If not, consider it as a cross edge between two L2 IDs in different chunks.
    Returns atomic edges and cross edges accordingly.
    """
    edge_layer = cg.get_cross_chunk_edges_layer([edge])[0]
    edge_parents = cg.get_parents(edge)

    # edge is within chunk
    if edge_layer == 1:
        return [edge_parents], {}
    # edge crosses atomic chunk boundary
    parent_1 = edge_parents[0]
    parent_2 = edge_parents[1]

    cross_edges_d = {}
    cross_edges_d[parent_1] = {edge_layer: edge}
    cross_edges_d[parent_2] = {edge_layer: edge[::-1]}
    atomic_edges = [[parent_1, parent_1], [parent_2, parent_2]]
    return (atomic_edges, cross_edges_d)


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
    atomic_edges, cross_edges_d = _analyze_atomic_edge(cg, edge)
    # TODO add read cross chunk edges method to client
