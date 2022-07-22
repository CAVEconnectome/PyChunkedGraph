"""
"plugin" to read agglomeration data provided by Ran Lu
"""

from collections import defaultdict
from itertools import product
from typing import Dict
from typing import Iterable
from typing import Tuple
from typing import Union
from binascii import crc32


import pandas as pd
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
from cloudfiles import CloudFiles

from .manager import IngestionManager
from .utils import postprocess_edge_data
from ..io.edges import put_chunk_edges
from ..io.components import put_chunk_components
from ..graph.utils import basetypes
from ..graph.edges import Edges
from ..graph.edges import EDGE_TYPES
from ..graph.chunks.utils import get_chunk_id

# see section below for description
CRC_LENGTH = 4
HEADER_LENGTH = 20

"""
Agglomeration data is now sharded.
Remap files and the region graph files are merged together
into bigger files with the following structure.

For example, "in_chunk_xxx_yyy.data" files are merge into a single "in_chunk_xxx.data",
and the reader needs to find out the range to extract the data for each chunk.

The layout of the new files is like this:

    byte 1-4: 'SO01' (version identifier)
    byte 5-12: Offset of the index information
    byte 13-20: Length of the index information (including crc32)
    byte 21-n: Payload data of the first chunk
    byte (n+1)-(n+4): Crc32 of the remap data of first chunk
    ...
    ...
    ...
    byte m-l: index data: (chunkid, offset, length)*k
    byte (l+1)-(l+4): Crc32 of the index data
"""


def read_raw_edge_data(imanager, coord) -> Dict:
    edge_dict = _collect_edge_data(imanager, coord)
    edge_dict = postprocess_edge_data(imanager, edge_dict)

    # flag to check if chunk has edges
    # avoid writing to cloud storage if there are no edges
    # unnecessary write operation
    no_edges = True
    chunk_edges = {}
    for edge_type in EDGE_TYPES:
        if not edge_dict[edge_type]:
            chunk_edges[edge_type] = Edges(np.array([]), np.array([]))
            continue
        sv_ids1 = edge_dict[edge_type]["sv1"]
        sv_ids2 = edge_dict[edge_type]["sv2"]
        areas = np.ones(len(sv_ids1))
        affinities = float("inf") * areas
        if not edge_type == EDGE_TYPES.cross_chunk:
            affinities = edge_dict[edge_type]["aff"]
            areas = edge_dict[edge_type]["area"]

        chunk_edges[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
        no_edges = no_edges and not sv_ids1.size
    if not no_edges and imanager.cg_meta.data_source.EDGES:
        put_chunk_edges(imanager.cg_meta.data_source.EDGES, coord, chunk_edges, 17)
    return chunk_edges


def _get_cont_chunk_coords(imanager, chunk_coord_a, chunk_coord_b):
    """Computes chunk coordinates that compute data between the named chunks."""
    diff = chunk_coord_a - chunk_coord_b
    dir_dim = np.where(diff != 0)[0]
    assert len(dir_dim) == 1
    dir_dim = dir_dim[0]

    chunk_coord_l = chunk_coord_a if diff[dir_dim] > 0 else chunk_coord_b
    c_chunk_coords = []
    for dx, dy, dz in product([0, -1], [0, -1], [0, -1]):
        if dz == dy == dx == 0:
            continue
        if [dx, dy, dz][dir_dim] == 0:
            continue

        c_chunk_coord = chunk_coord_l + np.array([dx, dy, dz])
        if imanager.cg_meta.is_out_of_bounds(c_chunk_coord):
            continue
        c_chunk_coords.append(c_chunk_coord)
    return c_chunk_coords


def _get_index(raw_data, in_chunk_or_agg_file=True):
    header = raw_data[:HEADER_LENGTH]
    idx_offset, idx_length = np.frombuffer(header[4:], dtype=np.uint64)
    idx_content = raw_data[int(idx_offset) : int(idx_offset + idx_length)]
    assert np.frombuffer(idx_content[-CRC_LENGTH:], dtype=np.uint32)[0] == crc32(
        idx_content[:-CRC_LENGTH]
    )
    idx_content = idx_content[:-CRC_LENGTH]

    dt = np.dtype([("chunkid", "u8"), ("offset", "u8"), ("size", "u8")])
    if in_chunk_or_agg_file is False:
        dt = np.dtype([("chunkid", "2u8"), ("offset", "u8"), ("size", "u8")])
    index_data = np.frombuffer(idx_content, dtype=dt)
    return index_data


def _crc_check(payload: bytes) -> None:
    payload_crc32 = np.frombuffer(payload[-CRC_LENGTH:], dtype=np.uint32)
    assert np.frombuffer(payload_crc32, dtype=np.uint32)[0] == crc32(
        payload[:-CRC_LENGTH]
    )


def _read_in_chunk_files(
    chunk_id: basetypes.NODE_ID,
    path: str,
    filenames: Iterable[str],
    edge_dtype: Iterable[Tuple],
):
    cf = CloudFiles(path)
    contents = cf.get(filenames, raw=True)

    data = []
    for f in contents:
        if f["error"] or f["content"] is None:
            continue
        raw = f["content"]
        index = _get_index(raw, in_chunk_or_agg_file=True)
        for chunk in index:
            if chunk["chunkid"] == chunk_id:
                payload = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
                _crc_check(payload)
                data.append(np.frombuffer(payload[:-CRC_LENGTH], dtype=edge_dtype))
    return data


def _read_between_or_fake_chunk_files(
    chunk_id: basetypes.NODE_ID,
    adjacent_id: basetypes.NODE_ID,
    path: str,
    filenames: Iterable[str],
    edge_dtype: Iterable[Tuple],
):
    cf = CloudFiles(path)
    contents = cf.get(filenames, raw=True)
    data = []
    for f in contents:
        if f["error"] or f["content"] is None:
            continue
        raw = f["content"]
        index = _get_index(raw, in_chunk_or_agg_file=False)
        for chunk in index:
            if chunk["chunkid"][0] == chunk_id and chunk["chunkid"][1] == adjacent_id:
                payload = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
                _crc_check(payload)
                data.append(np.frombuffer(payload[:-CRC_LENGTH], dtype=edge_dtype))
            if chunk["chunkid"][0] == adjacent_id and chunk["chunkid"][1] == chunk_id:
                payload = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
                _crc_check(payload)
                this_dtype = [edge_dtype[1], edge_dtype[0]] + edge_dtype[2:]
                data.append(np.frombuffer(payload[:-CRC_LENGTH], dtype=this_dtype))
    return data


def _collect_edge_data(imanager: IngestionManager, chunk_coord):
    """Loads edge for single chunk."""
    cg_meta = imanager.cg_meta
    edge_dtype = cg_meta.edge_dtype
    subfolder = "chunked_rg"
    path = f"{imanager.config.AGGLOMERATION}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = get_chunk_id(cg_meta, layer=1, x=x, y=y, z=z)

    edge_data = defaultdict(list)
    in_fnames = []
    x, y, z = chunk_coord
    for _x, _y, _z in product([x - 1, x], [y - 1, y], [z - 1, z]):
        if cg_meta.is_out_of_bounds(np.array([_x, _y, _z])):
            continue
        filename = f"in_chunk_0_{_x}_{_y}_{_z}.data"
        in_fnames.append(filename)

    edge_data[EDGE_TYPES.in_chunk] = _read_in_chunk_files(
        chunk_id,
        path,
        in_fnames,
        edge_dtype,
    )
    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adjacent_coord = chunk_coord + diff
            x, y, z = adjacent_coord
            adjacent_id = get_chunk_id(cg_meta, layer=1, x=x, y=y, z=z)
            if cg_meta.is_out_of_bounds(adjacent_coord):
                continue

            cont_coords = _get_cont_chunk_coords(imanager, chunk_coord, adjacent_coord)
            bt_fnames = []
            cx_fnames = []
            for c_chunk_coord in cont_coords:
                x, y, z = c_chunk_coord
                filename = f"between_chunks_0_{x}_{y}_{z}.data"
                bt_fnames.append(filename)

                # EDGES FROM CUTS OF SVS
                filename = f"fake_0_{x}_{y}_{z}.data"
                cx_fnames.append(filename)

            for edge_type, fnames in [
                (EDGE_TYPES.between_chunk, bt_fnames),
                (EDGE_TYPES.cross_chunk, cx_fnames),
            ]:
                _data = _read_between_or_fake_chunk_files(
                    chunk_id,
                    adjacent_id,
                    path,
                    fnames,
                    edge_dtype,
                )
                edge_data[edge_type].extend(_data)

    for k in EDGE_TYPES:
        if not edge_data[k]:
            continue
        edge_data[k] = rfn.stack_arrays(edge_data[k], usemask=False)
        edge_data_df = pd.DataFrame(edge_data[k])
        edge_data_dfg = (
            edge_data_df.groupby(["sv1", "sv2"]).aggregate(np.sum).reset_index()
        )
        edge_data[k] = edge_data_dfg.to_records()
    return edge_data


def get_active_edges(imanager: IngestionManager, coord, edges_d, mapping):
    active_edges_flag_d, isolated_ids = define_active_edges(edges_d, mapping)
    chunk_edges_active = {}
    pseudo_isolated_ids = [isolated_ids]
    for edge_type in EDGE_TYPES:
        edges = edges_d[edge_type]

        active = (
            np.ones(len(edges), dtype=bool)
            if edge_type == EDGE_TYPES.cross_chunk
            else active_edges_flag_d[edge_type]
        )

        sv_ids1 = edges.node_ids1[active]
        sv_ids2 = edges.node_ids2[active]
        affinities = edges.affinities[active]
        areas = edges.areas[active]
        chunk_edges_active[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
        # assume all ids within the chunk are isolated
        # to make sure all end up in connected components
        pseudo_isolated_ids.append(edges.node_ids1)
        if edge_type == EDGE_TYPES.in_chunk:
            pseudo_isolated_ids.append(edges.node_ids2)

    return chunk_edges_active, np.unique(np.concatenate(pseudo_isolated_ids))


def define_active_edges(edge_dict, mapping) -> Union[Dict, np.ndarray]:
    """Labels edges as within or across segments and extracts isolated ids
    :return: dict of np.ndarrays, np.ndarray
        bool arrays; True: connected (within same segment)
        isolated node ids
    """
    mapping_vec = np.vectorize(lambda k: mapping.get(k, -1))
    active = {}
    isolated = [[]]
    for k in edge_dict:
        if len(edge_dict[k].node_ids1) > 0:
            agg_id_1 = mapping_vec(edge_dict[k].node_ids1)
        else:
            assert len(edge_dict[k].node_ids2) == 0
            active[k] = np.array([], dtype=bool)
            continue

        agg_id_2 = mapping_vec(edge_dict[k].node_ids2)
        active[k] = agg_id_1 == agg_id_2
        # Set those with two -1 to False
        agg_1_m = agg_id_1 == -1
        agg_2_m = agg_id_2 == -1
        active[k][agg_1_m] = False

        isolated.append(edge_dict[k].node_ids1[agg_1_m])
        if k == EDGE_TYPES.in_chunk:
            isolated.append(edge_dict[k].node_ids2[agg_2_m])
    return active, np.unique(np.concatenate(isolated).astype(basetypes.NODE_ID))


def read_raw_agglomeration_data(imanager: IngestionManager, chunk_coord: np.ndarray):
    """
    Collects agglomeration information & builds connected component mapping
    """
    cg_meta = imanager.cg_meta
    subfolder = "remap"
    path = f"{imanager.config.AGGLOMERATION}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = get_chunk_id(cg_meta, layer=1, x=x, y=y, z=z)

    filenames = []
    chunk_ids = []
    for mip_level in range(0, int(cg_meta.layer_count - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}.data")
        chunk_ids.append(chunk_id)

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adjacent_coord = chunk_coord + diff
            x, y, z = adjacent_coord
            adjacent_id = get_chunk_id(cg_meta, layer=1, x=x, y=y, z=z)

            for mip_level in range(0, int(cg_meta.layer_count - 1)):
                x, y, z = np.array(adjacent_coord / 2 ** mip_level, dtype=int)
                filenames.append(f"done_{mip_level}_{x}_{y}_{z}.data")
                chunk_ids.append(adjacent_id)

    edges_list = _read_agg_files(filenames, chunk_ids, path)
    G = nx.Graph()
    G.add_edges_from(np.concatenate(edges_list))
    mapping = {}
    components = list(nx.connected_components(G))
    for i_cc, cc in enumerate(components):
        cc = list(cc)
        mapping.update(dict(zip(cc, [i_cc] * len(cc))))

    if mapping and cg_meta.data_source.COMPONENTS:
        put_chunk_components(cg_meta.data_source.COMPONENTS, components, chunk_coord)
    return mapping


def _read_agg_files(filenames, chunk_ids, path):
    from ..graph.types import empty_2d

    cf = CloudFiles(path)
    contents_d = cf.get(set(filenames), raw=True, return_dict=True)

    edge_list = [empty_2d]
    for fname, chunk_id in zip(filenames, chunk_ids):
        raw = contents_d[fname]
        if raw is None:
            continue
        edges = None
        index = _get_index(raw, in_chunk_or_agg_file=True)
        for chunk in index:
            if chunk["chunkid"] == chunk_id:
                data = raw[chunk["offset"] : chunk["offset"] + chunk["size"]]
                data_ = data[:-CRC_LENGTH]
                edges = np.frombuffer(data_, dtype=basetypes.NODE_ID).reshape(-1, 2)
                if edges is not None:
                    edge_list.append(edges)
                break
    return edge_list
