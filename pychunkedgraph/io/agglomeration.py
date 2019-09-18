import json
from typing import Dict

import numpy as np
from cloudvolume.storage import SimpleStorage

from .protobuf.chunkMapping_pb2 import ChunkMappingMsg
from ..backend.utils import basetypes


def serialize(mapping: Dict) -> ChunkMappingMsg:
    supervoxels = np.array(list(mapping.keys()), dtype=basetypes.NODE_ID)
    components = np.array(list(mapping.values()), dtype=int)
    mapping_message = ChunkMappingMsg()
    mapping_message.supervoxels = supervoxels.tobytes()
    mapping_message.components = components.tobytes()
    return mapping_message


def deserialize(mapping_message: ChunkMappingMsg) -> Dict:
    supervoxels = np.frombuffer(mapping_message.supervoxels, basetypes.NODE_ID)
    components = np.frombuffer(mapping_message.components, basetypes.NODE_ID)
    return dict(zip(supervoxels, components))


def put_chunk_agglomeration(agglomeration_dir, mapping, chunk_coord) -> None:
    # filename format - mapping_x_y_z.serliazation
    mapping_message = serialize(mapping)
    file_name = f"mapping_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(agglomeration_dir) as storage:
        storage.put_file(
            file_path=file_name,
            content=mapping_message.SerializeToString(),
            compress="gzip",
            cache_control="no-cache",
        )


def get_chunk_agglomeration(agglomeration_dir, chunk_coord) -> Dict:
    # filename format - mapping_x_y_z.serliazation
    file_name = f"mapping_{'_'.join(str(coord) for coord in chunk_coord)}.proto"
    with SimpleStorage(agglomeration_dir) as storage:
        content = storage.get_file(file_name)
        if not content:
            return {}
        mapping_message = ChunkMappingMsg()
        mapping_message.ParseFromString(content)
        return deserialize(mapping_message)

