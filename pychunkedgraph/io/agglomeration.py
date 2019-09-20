import json
from typing import Dict

import numpy as np
from cloudvolume.storage import SimpleStorage

from .protobuf.chunkComponents_pb2 import ChunkComponentsMsg
from ..backend.utils import basetypes


def serialize(mapping: Dict) -> ChunkComponentsMsg:
    supervoxels = np.array(list(mapping.keys()), dtype=basetypes.NODE_ID)
    components = np.array(list(mapping.values()), dtype=int)
    components_message = ChunkComponentsMsg()
    components_message.supervoxels = supervoxels.tobytes()
    components_message.components = components.tobytes()
    return components_message


def deserialize(components_message: ChunkComponentsMsg) -> Dict:
    supervoxels = np.frombuffer(components_message.supervoxels, basetypes.NODE_ID)
    components = np.frombuffer(components_message.components, basetypes.NODE_ID)
    return dict(zip(supervoxels, components))


def put_chunk_components(agglomeration_dir, mapping, chunk_coord) -> None:
    # filename format - components_x_y_z.serliazation
    components_message = serialize(mapping)
    file_name = f"components_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(agglomeration_dir) as storage:
        storage.put_file(
            file_path=file_name,
            content=components_message.SerializeToString(),
            compress="gzip",
            cache_control="no-cache",
        )


def get_chunk_components(agglomeration_dir, chunk_coord) -> Dict:
    # filename format - components_x_y_z.serliazation
    file_name = f"components_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(agglomeration_dir) as storage:
        content = storage.get_file(file_name)
        if not content:
            return {}
        components_message = ChunkComponentsMsg()
        components_message.ParseFromString(content)
        return deserialize(components_message)

