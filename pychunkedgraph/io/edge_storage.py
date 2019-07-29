"""
Functions for reading and writing edges 
to secondary storage with CloudVolume
"""

import os
from typing import List, Dict, Tuple

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from cloudvolume.storage import SimpleStorage
from .protobuf.chunkEdges_pb2 import Edges


def _decompress_edges(content: bytes):
    """
    :param content: zstd compressed bytes
    :type bytes:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    edgesMessage = Edges()
    zstdDecompressorObj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstdDecompressorObj.decompress(content)
    edgesMessage.ParseFromString(file_content)

    edges = np.frombuffer(edgesMessage.edgeList).reshape(-1, 2)
    affinities = np.frombuffer(edgesMessage.affinities, dtype="<f4")
    areas = np.frombuffer(edgesMessage.areas, dtype="<u8")
    return edges, affinities, areas


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunks_coordinates:
    :type np.ndarray:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # this is just for testing
    edges_dir = os.environ.get(
        "EDIR", "gs://akhilesh-test/edges/fly_playground/bbox-102_51_5-110_59_9"
    )
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # TODO change filename format
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"chunk_{chunk_str}_zstd_level_17_proto.data")

    edges = np.array([], dtype=np.uint64).reshape(0, 2)
    affinities = np.array([], dtype=np.float32)
    areas = np.array([], dtype=np.uint64)

    st = SimpleStorage(edges_dir)
    if cv_threads > 1:
        st = Storage(edges_dir, n_threads=cv_threads)

    files = []
    with st:
        files = st.get_files(fnames)
        for _file in files:
            if not _file["content"]:
                continue
            _edges, _affinities, _areas = _decompress_edges(_file["content"])
            edges = np.concatenate([edges, _edges])
            affinities = np.concatenate([affinities, _affinities])
            areas = np.concatenate([areas, _areas])

    return edges, affinities, areas


def put_chunk_edges(
    edges_dir: str,
    chunk_coordinates: np.ndarray,
    edges: np.ndarray,
    affinities: np.ndarray,
    areas: np.ndarray,
    compression_level: int,
) -> None:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunk_coordinates: chunk coords x,y,z
    :type np.ndarray:
    :param edges: np.array of [supervoxel1, supervoxel2]
    :type np.ndarray:
    :param affinities:
    :type np.ndarray:
    :param areas:
    :type np.ndarray:
    :param compression_level: zstandard compression level (1-22, higher - better ratio)
    :type int:
    :return None:
    """
    edgesMessage = Edges()
    edgesMessage.edgeList = edges.tobytes()
    edgesMessage.affinities = affinities.tobytes()
    edgesMessage.areas = areas.tobytes()

    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_proto = cctx.compress(edgesMessage.SerializeToString())

    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)
    # filename format - edges_x_y_z.serialization.compression

    file = f"edges_{chunk_str}.proto.zst"
    with Storage(edges_dir) as st:
        st.put_file(
            file_path=file,
            content=compressed_proto,
            compress=None,
            cache_control="no-cache",
        )
