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
    :return: Tuple[edges:np.array[np.uint64, np.uint64],
                   areas:np.array[np.uint64]
                   affinities: np.array[np.float64]]
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
    :param: chunks_coordinates np.array of chunk coordinates
    :return: tuple of edge infos (edges, affinities, areas)
    """
    edges_dir = os.environ.get(
        "EDIR", "gs://akhilesh-test/edges/fly_playground/bbox-102_51_5-110_59_9"
    )
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
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
    chunk_str: str,
    edges: np.ndarray,
    affinities: np.ndarray,
    areas: np.ndarray,
    edges_dir: str,
    compression_level: int,
) -> None:
    """
    :param: chunk_str - chunk coords in format x_y_z
    :type: str
    :param: edges - (supervoxel1, supervoxel2)
    :type: np.ndarray
    :param: affinities
    :type: np.ndarray
    :param: areas
    :type: np.ndarray
    :param: edges_dir - google cloud storage path
    :type: str
    :param: compression_level - for zstandard (1-22, higher - better ratio)
    :type: int
    """
    edgesMessage = Edges()
    edgesMessage.edgeList = edges.tobytes()
    edgesMessage.affinities = affinities.tobytes()
    edgesMessage.areas = areas.tobytes()

    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_proto = cctx.compress(edgesMessage.SerializeToString())

    # filename - "chunk_" + chunk_coords + compression_tool + serialization_method

    file = f"chunk_{chunk_str}_zstd_proto.data"
    with Storage(edges_dir) as st:
        st.put_file(
            file_path=file,
            content=compressed_proto,
            compress=None,
            cache_control="no-cache",
        )
