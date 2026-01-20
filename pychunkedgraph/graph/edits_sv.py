"""
Supervoxel splitting and managing new IDs.
"""

from typing import Iterable

import numpy as np

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.utils import basetypes
from pychunkedgraph.graph.attributes import Hierarchy
from pychunkedgraph.graph.utils.serializers import serialize_uint64


def split_supervoxel(
    cg: ChunkedGraph, supervoxel_id: basetypes.NODE_ID
) -> Iterable[basetypes.NODE_ID]:
    """
    Lookup coordinates of given supervoxel in segmentation.
    Split it and update the coordinates with new IDs.
    Return new IDs.
    """


def copy_parents_and_create_lineage(
    cg: ChunkedGraph, old_id: basetypes.NODE_ID, new_ids: Iterable[basetypes.NODE_ID]
) -> list:
    """
    Copy parents column from `old_id` to each of `new_ids`.
      This makes it easy to get old hierarchy with `new_ids` using an older timestamp.
    Link `old_id` and `new_ids` to create a lineage at supervoxel layer.
    Returns a list of mutations to be persisted.
    """
    result = []
    parent_cells = cg.client.read_node(old_id, properties=Hierarchy.Parent)

    for new_id in new_ids:
        val_dict = {
            Hierarchy.FormerIdentity: np.array([old_id], dtype=basetypes.NODE_ID)
        }
        result.append(cg.client.mutate_row(serialize_uint64(new_id), val_dict))

        for cell in parent_cells:
            result.append(
                cg.client.mutate_row(
                    serialize_uint64(new_id),
                    {Hierarchy.Parent: cell.value},
                    time_stamp=cell.timestamp,
                )
            )

    val_dict = {Hierarchy.NewIdentity: np.array(new_ids, dtype=basetypes.NODE_ID)}
    result.append(cg.client.mutate_row(serialize_uint64(old_id), val_dict))
    return result
