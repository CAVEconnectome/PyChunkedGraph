from typing import Union

import numpy as np

from ..graph import ChunkedGraph
from ..graph.operation import GraphEditOperation
from ..graph.operation import MergeOperation
from ..graph.operation import MulticutOperation
from ..graph.operation import SplitOperation
from ..app.app_utils import handle_supervoxel_id_lookup

USER_ID = "debug"


def _parse_merge_payload(
    cg: ChunkedGraph, user_id: str, payload: dict
) -> MergeOperation:
    pass


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
    return MulticutOperation(
        cg,
        user_id=user_id,
        source_ids=source_ids,
        sink_ids=sink_ids,
        source_coords=source_coords,
        sink_coords=sink_coords,
        bbox_offset=bb_offset,
        path_augment=True,
        disallow_isolating_cut=True,
    )


def get_operation_from_request_payload(
    cg: ChunkedGraph,
    payload: dict,
    split: bool,
    *,
    mincut: bool = True,
    user_id: str = None,
) -> GraphEditOperation:
    if user_id is None:
        user_id = USER_ID
    if split:
        return _parse_split_payload(cg, user_id, payload, mincut=mincut)
