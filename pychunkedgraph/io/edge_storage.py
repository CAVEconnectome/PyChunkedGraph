"""
Functions for reading and writing edges 
to (slow) storage with CloudVolume
"""

from typing import List, Dict, Tuple, Union

import numpy as np
import zstandard as zstd

from cloudvolume import Storage
from cloudvolume.storage import SimpleStorage

from ..backend.utils import basetypes
from .protobuf.chunkEdges_pb2 import EdgesMsg, ChunkEdgesMsg


def _decompress_edges(content: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param content: zstd compressed bytes
    :type bytes:
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    def _get_edges(edges_message: EdgesMsg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        supervoxel_ids1 = np.frombuffer(edges_message.node_ids1, basetypes.NODE_ID)
        supervoxel_ids2 = np.frombuffer(edges_message.node_ids2, basetypes.NODE_ID)

        edges = np.column_stack((supervoxel_ids1, supervoxel_ids2))
        affinities = np.frombuffer(edges_message.affinities, basetypes.EDGE_AFFINITY)
        areas = np.frombuffer(edges_message.areas, basetypes.EDGE_AREA)
        return edges, affinities, areas

    chunk_edges = ChunkEdgesMsg()
    zstd_decompressor_obj = zstd.ZstdDecompressor().decompressobj()
    file_content = zstd_decompressor_obj.decompress(content)
    chunk_edges.ParseFromString(file_content)

    # in, between and cross
    in_edges, in_affinities, in_areas = _get_edges(chunk_edges.in_chunk)
    bt_edges, bt_affinities, bt_areas = _get_edges(chunk_edges.between_chunk)
    cx_edges, cx_affinities, cx_areas = _get_edges(chunk_edges.cross_chunk)

    edges = np.concatenate([in_edges, bt_edges, cx_edges])
    affinities = np.concatenate([in_affinities, bt_affinities, cx_affinities])
    areas = np.concatenate([in_areas, bt_areas, cx_areas])

    return edges, affinities, areas


def get_chunk_edges(
    edges_dir: str, chunks_coordinates: List[np.ndarray], cv_threads: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param edges_dir: cloudvolume storage path
    :type str:    
    :param chunks_coordinates: list of chunk coords for which to load edges
    :type List[np.ndarray]:
    :param cv_threads: cloudvolume storage client thread count
    :type int:     
    :return: edges, affinities, areas
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    fnames = []
    for chunk_coords in chunks_coordinates:
        chunk_str = "_".join(str(coord) for coord in chunk_coords)
        # filename format - edges_x_y_z.serialization.compression
        fnames.append(f"edges_{chunk_str}.proto.zst")

    edges = np.array([], basetypes.NODE_ID).reshape(0, 2)
    affinities = np.array([], basetypes.EDGE_AFFINITY)
    areas = np.array([], basetypes.EDGE_AREA)

    st = (
        Storage(edges_dir, n_threads=cv_threads)
        if cv_threads > 1
        else SimpleStorage(edges_dir)
    )

    files = []
    with st:
        files = st.get_files(fnames)
        for _file in files:
            # cv error
            if _file["error"]:
                raise ValueError(_file["error"])
            # empty chunk
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
    chunk_edges_raw,
    compression_level: int,
) -> None:
    """
    :param edges_dir: cloudvolume storage path
    :type str:
    :param chunk_coordinates: chunk coords x,y,z
    :type np.ndarray:
    :param chunk_edges_raw: chunk_edges_raw with keys "in", "cross", "between"
    :type dict:
    :param compression_level: zstandard compression level (1-22, higher - better ratio)
    :type int:
    :return None:
    """

    def _get_edges(edge_type: str) -> EdgesMsg:
        edges = chunk_edges_raw[edge_type]
        edges_proto = EdgesMsg()
        edges_proto.node_ids1 = edges.node_ids1.astype(basetypes.NODE_ID).tobytes()
        edges_proto.node_ids2 = edges.node_ids2.astype(basetypes.NODE_ID).tobytes()
        edges_proto.affinities = edges.affinities.astype(
            basetypes.EDGE_AFFINITY
        ).tobytes()
        edges_proto.areas = edges.areas.astype(basetypes.EDGE_AREA).tobytes()

        return edges_proto

    chunk_edges = ChunkEdgesMsg()
    chunk_edges.in_chunk.CopyFrom(_get_edges("in"))
    chunk_edges.between_chunk.CopyFrom(_get_edges("between"))
    chunk_edges.cross_chunk.CopyFrom(_get_edges("cross"))

    cctx = zstd.ZstdCompressor(level=compression_level)
    chunk_str = "_".join(str(coord) for coord in chunk_coordinates)

    # filename format - edges_x_y_z.serialization.compression
    file = f"edges_{chunk_str}.proto.zst"
    with Storage(edges_dir) as st:
        st.put_file(
            file_path=file,
            content=cctx.compress(chunk_edges.SerializeToString()),
            compress=None,
            cache_control="no-cache",
        )
