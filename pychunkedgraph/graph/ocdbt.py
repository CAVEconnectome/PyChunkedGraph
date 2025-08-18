import os
import numpy as np
import tensorstore as ts

from pychunkedgraph.graph import ChunkedGraph

OCDBT_SEG_COMPRESSION_LEVEL = 22


def get_seg_source_and_destination_ocdbt(
    cg: ChunkedGraph, create: bool = False
) -> tuple:
    src_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": cg.meta.data_source.WATERSHED,
    }
    src = ts.open(src_spec).result()
    schema = src.schema

    ocdbt_path = os.path.join(cg.meta.data_source.WATERSHED, "ocdbt", "base")
    dst_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "ocdbt",
            "base": ocdbt_path,
            "config": {
                "compression": {"id": "zstd", "level": OCDBT_SEG_COMPRESSION_LEVEL},
            },
        },
    }

    dst = ts.open(
        dst_spec,
        create=create,
        rank=schema.rank,
        dtype=schema.dtype,
        codec=schema.codec,
        domain=schema.domain,
        shape=schema.shape,
        chunk_layout=schema.chunk_layout,
        dimension_units=schema.dimension_units,
        delete_existing=create,
    ).result()
    return (src, dst)


def copy_ws_chunk(cg: ChunkedGraph, coords: list, source, destination):
    coords = np.array(coords, dtype=int)
    vx_start = coords * cg.meta.graph_config.CHUNK_SIZE
    vx_end = vx_start + cg.meta.graph_config.CHUNK_SIZE
    xE, yE, zE = cg.meta.voxel_bounds[:, 1]

    x0, y0, z0 = vx_start
    x1, y1, z1 = vx_end
    x1 = min(x1, xE)
    y1 = min(y1, yE)
    z1 = min(z1, zE)

    data = source[x0:x1, y0:y1, z0:z1].read().result()
    destination[x0:x1, y0:y1, z0:z1].write(data).result()
