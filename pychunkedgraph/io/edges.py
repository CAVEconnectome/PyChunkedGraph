# pylint: disable=invalid-name, missing-docstring
"""
Functions for reading and writing edges from cloud storage.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import zstandard as zstd
from cloudfiles import CloudFiles

from .protobuf.chunkEdges_pb2 import EdgesMsg
from .protobuf.chunkEdges_pb2 import ChunkEdgesMsg
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.utils import basetypes
from ..graph.edges.utils import concatenate_chunk_edges


def serialize(edges: Edges) -> EdgesMsg:
    edges_proto = EdgesMsg()
    edges_proto.node_ids1 = edges.node_ids1.astype(basetypes.NODE_ID).tobytes()
    edges_proto.node_ids2 = edges.node_ids2.astype(basetypes.NODE_ID).tobytes()
    edges_proto.affinities = edges.affinities.astype(basetypes.EDGE_AFFINITY).tobytes()
    edges_proto.areas = edges.areas.astype(basetypes.EDGE_AREA).tobytes()
    return edges_proto


def deserialize(edges_message: EdgesMsg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sv_ids1 = np.frombuffer(edges_message.node_ids1, basetypes.NODE_ID)
    sv_ids2 = np.frombuffer(edges_message.node_ids2, basetypes.NODE_ID)
    affinities = np.frombuffer(edges_message.affinities, basetypes.EDGE_AFFINITY)
    areas = np.frombuffer(edges_message.areas, basetypes.EDGE_AREA)
    return Edges(sv_ids1, sv_ids2, affinities=affinities, areas=areas)


def _decompress_one(zdc, content):
    # zdc.decompress needs the content size in the frame header, which is what
    # put_chunk_edges writes (and what multi_decompress_to_buffer also required).
    # decompressobj streams and needs no size, so it covers any frame lacking it.
    try:
        return zdc.decompress(content)
    except zstd.ZstdError:
        return zdc.decompressobj().decompress(content)


def _decompress(compressed: List[bytes], n_threads: int) -> List[bytes]:
    """Decompress chunk blobs, in parallel when n_threads > 1.

    Replaces multi_decompress_to_buffer, which zstandard removed in 0.23. A thread pool
    matches it because decompression releases the GIL: measured on zstandard 0.21.0 with
    192 MB across 64 blobs and 4 threads, 41.7 ms for the pool vs 42.5 ms for
    multi_decompress_to_buffer, against 164.8 ms serial. This keeps that ~4x without
    depending on a specific zstandard version.

    ZstdDecompressor is not thread-safe -- sharing one across threads silently produces
    corrupt output rather than raising -- so each worker keeps its own.
    """
    if n_threads <= 1 or len(compressed) < 2:
        zdc = zstd.ZstdDecompressor()
        return [_decompress_one(zdc, content) for content in compressed]

    local = threading.local()

    def _one(content):
        zdc = getattr(local, "zdc", None)
        if zdc is None:
            zdc = local.zdc = zstd.ZstdDecompressor()
        return _decompress_one(zdc, content)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        return list(pool.map(_one, compressed))


def _parse_edges(compressed: List[bytes], sorted_svs: np.ndarray = None) -> List[Dict]:
    result = []
    if(len(compressed) == 0):
        return result
    try:
        n_threads = int(os.environ.get("ZSTD_THREADS", 1))
    except ValueError:
        n_threads = 1

    decompressed = _decompress(compressed, n_threads)

    for content in decompressed:
        chunk_edges = ChunkEdgesMsg()
        chunk_edges.ParseFromString(memoryview(content))
        edges_dict = {}
        edges_dict[EDGE_TYPES.in_chunk] = deserialize(chunk_edges.in_chunk)
        edges_dict[EDGE_TYPES.between_chunk] = deserialize(chunk_edges.between_chunk)
        edges_dict[EDGE_TYPES.cross_chunk] = deserialize(chunk_edges.cross_chunk)
        if sorted_svs is not None:
            for edge_type, edges in edges_dict.items():
                edges_dict[edge_type] = edges.filter_touching(sorted_svs)
        result.append(edges_dict)
    return result


try:
    EDGES_BATCH_SIZE = int(os.environ.get("PCG_EDGES_BATCH_SIZE", 64))
except ValueError:
    EDGES_BATCH_SIZE = 64


def get_chunk_edges(
    edges_dir: str,
    chunks_coordinates: List[np.ndarray],
    supervoxels: np.ndarray = None,
    batch_size: int = None,
) -> Dict:
    """Read edges from GCS.

    :param supervoxels: optional supervoxel ids of the object being queried. When given,
        each chunk is filtered to edges touching one of them before anything is retained,
        so peak memory tracks the size of the object rather than the total edge content of
        the chunks it spans. ``None`` reads every edge (the ingest path relies on this).
    :param batch_size: how many chunk files to fetch and decompress at a time. Bounds the
        transient buffers to the batch instead of the whole request, so a query spanning
        many chunks costs no more per moment than one spanning a few. Defaults to
        PCG_EDGES_BATCH_SIZE (64).
    """
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    # sort once here rather than per chunk inside the parse loop
    sorted_svs = None
    if supervoxels is not None:
        sorted_svs = np.unique(np.asarray(supervoxels, dtype=basetypes.NODE_ID))

    if batch_size is None:
        batch_size = EDGES_BATCH_SIZE
    batch_size = max(1, batch_size)

    cf = CloudFiles(edges_dir, num_threads=4)
    # Accumulate the per-chunk dicts batch by batch. Each batch's compressed and
    # decompressed buffers are released before the next is fetched; only the filtered
    # survivors are carried forward, so peak tracks batch_size rather than len(fnames).
    parsed = []
    for start in range(0, len(fnames), batch_size):
        files = cf.get(fnames[start : start + batch_size], raw=True)
        compressed = []
        for f in files:
            if not f["content"]:
                continue
            compressed.append(f["content"])
        del files
        parsed.extend(_parse_edges(compressed, sorted_svs))
        del compressed
    return concatenate_chunk_edges(parsed)


def put_chunk_edges(
    edges_dir: str, chunk_coordinates: np.ndarray, edges_d, compression_level: int
) -> None:
    """Write edges to GCS."""
    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.in_chunk]))
    chunk_edges.between_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.between_chunk]))
    chunk_edges.cross_chunk.CopyFrom(serialize(edges_d[EDGE_TYPES.cross_chunk]))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    filename = f"edges_{chunk_str}.proto.zst"
    cf = CloudFiles(edges_dir)
    cf.put(
        filename,
        content=cctx.compress(chunk_edges.SerializeToString()),
        compress=None,
        cache_control="no-cache",
    )
