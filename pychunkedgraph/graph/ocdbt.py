import os
import numpy as np
import tensorstore as ts

OCDBT_SEG_COMPRESSION_LEVEL = 17


def get_seg_source_and_destination_ocdbt(ws_path: str, create: bool = False) -> tuple:
    src_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": ws_path,
    }
    src = ts.open(src_spec).result()
    schema = src.schema

    ocdbt_path = os.path.join(ws_path, "ocdbt", "base")
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


def copy_ws_chunk(
    source,
    destination,
    chunk_size: tuple,
    coords: list,
    voxel_bounds: np.ndarray,
):
    coords = np.array(coords, dtype=int)
    chunk_size = np.array(chunk_size, dtype=int)
    vx_start = coords * chunk_size + voxel_bounds[:, 0]
    vx_end = vx_start + chunk_size
    xE, yE, zE = voxel_bounds[:, 1]

    x0, y0, z0 = vx_start
    x1, y1, z1 = vx_end
    x1 = min(x1, xE)
    y1 = min(y1, yE)
    z1 = min(z1, zE)

    data = source[x0:x1, y0:y1, z0:z1].read().result()
    destination[x0:x1, y0:y1, z0:z1].write(data).result()
