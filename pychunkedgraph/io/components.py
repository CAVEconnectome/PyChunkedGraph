import json
from typing import Dict, Iterable

import numpy as np
from cloudvolume.storage import SimpleStorage

from .protobuf.chunkComponents_pb2 import ChunkComponentsMsg
from ..graph.utils import basetypes


def serialize(connected_components: Iterable) -> ChunkComponentsMsg:
    components = []
    for component in list(connected_components):
        component = np.array(list(component), dtype=basetypes.NODE_ID)
        components.append(np.array([len(component)], dtype=basetypes.NODE_ID))
        components.append(component)
    components_message = ChunkComponentsMsg()
    components_message.components[:] = np.concatenate(components)
    return components_message


def deserialize(components_message: ChunkComponentsMsg) -> Dict:
    mapping = {}
    components = np.array(components_message.components, basetypes.NODE_ID)
    idx = 0
    n_components = 0
    while idx < components.size:
        component_size = int(components[idx])
        start = idx + 1
        component = components[start : start + component_size]
        mapping.update(dict(zip(component, [n_components] * component_size)))
        idx += component_size + 1
        n_components += 1
    return mapping


def put_chunk_components(components_dir, components, chunk_coord) -> None:
    # filename format - components_x_y_z.serliazation
    components_message = serialize(components)
    file_name = f"components_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(components_dir) as storage:
        storage.put_file(
            file_path=file_name,
            content=components_message.SerializeToString(),
            compress="gzip",
            cache_control="no-cache",
        )


def get_chunk_components(components_dir, chunk_coord) -> Dict:
    # filename format - components_x_y_z.serliazation
    file_name = f"components_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(components_dir) as storage:
        content = storage.get_file(file_name)
        if not content:
            return {}
        components_message = ChunkComponentsMsg()
        components_message.ParseFromString(content)
        return deserialize(components_message)

