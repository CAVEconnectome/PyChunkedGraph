from typing import Union
from typing import Tuple

import numpy as np

from ..graph import ChunkedGraph
from ..graph.operation import GraphEditOperation
from ..graph.operation import MergeOperation
from ..graph.operation import MulticutOperation
from ..graph.operation import SplitOperation
from ..app.app_utils import handle_supervoxel_id_lookup

USER_ID = "debug"


def _parse_merge_payload(
    cg: ChunkedGraph, user_id: str, payload: list
) -> MergeOperation:

    node_ids = []
    coords = []
    for node in payload:
        node_ids.append(node[0])
        coords.append(np.array(node[1:]) / cg.segmentation_resolution)

    atomic_edge = handle_supervoxel_id_lookup(cg, coords, node_ids)
    chunk_coord_delta = cg.get_chunk_coordinates(
        atomic_edge[0]
    ) - cg.get_chunk_coordinates(atomic_edge[1])
    if np.any(np.abs(chunk_coord_delta) > 3):
        raise ValueError("Chebyshev distance exceeded allowed maximum.")

    return (
        node_ids,
        atomic_edge,
        MergeOperation(
            cg,
            user_id=user_id,
            added_edges=np.array(atomic_edge, dtype=np.uint64),
            source_coords=coords[:1],
            sink_coords=coords[1:],
        ),
    )


def _parse_split_payload(
    cg: ChunkedGraph, user_id: str, payload: dict, mincut: bool = True
) -> Union[SplitOperation, MulticutOperation]:
    node_idents = []
    node_ident_map = {
        "sources": 0,
        "sinks": 1,
    }
    coords = []
    node_ids = []

    for k in ["sources", "sinks"]:
        for node in payload[k]:
            node_ids.append(node[0])
            coords.append(np.array(node[1:]) / cg.segmentation_resolution)
            node_idents.append(node_ident_map[k])

    node_ids = np.array(node_ids, dtype=np.uint64)
    coords = np.array(coords)
    node_idents = np.array(node_idents)
    sv_ids = handle_supervoxel_id_lookup(cg, coords, node_ids)

    source_ids = sv_ids[node_idents == 0]
    sink_ids = sv_ids[node_idents == 1]
    source_coords = coords[node_idents == 0]
    sink_coords = coords[node_idents == 1]

    bb_offset = (240, 240, 24)
    return (
        source_ids,
        sink_ids,
        MulticutOperation(
            cg,
            user_id=user_id,
            source_ids=source_ids,
            sink_ids=sink_ids,
            source_coords=source_coords,
            sink_coords=sink_coords,
            bbox_offset=bb_offset,
            path_augment=True,
            disallow_isolating_cut=True,
        ),
    )


def get_operation_from_request_payload(
    cg: ChunkedGraph,
    payload: Union[list, dict],
    split: bool,
    *,
    mincut: bool = True,
    user_id: str = None,
) -> Tuple[np.ndarray, np.ndarray, GraphEditOperation]:
    if user_id is None:
        user_id = USER_ID
    if split:
        return _parse_split_payload(cg, user_id, payload, mincut=mincut)
    return _parse_merge_payload(cg, user_id, payload)
