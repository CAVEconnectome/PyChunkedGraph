"""
"plugin" to read agglomeration data provided by Ran Lu
"""

import time
import json
from collections import defaultdict
from collections import Counter
from itertools import product
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Sequence

import pandas as pd
import cloudvolume
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
import zstandard as zstd

from .manager import IngestionManager
from .utils import postprocess_edge_data
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.components import get_chunk_components
from ..io.components import put_chunk_components
from ..backend import ChunkedGraphMeta
from ..backend.utils import basetypes
from ..backend.edges import Edges
from ..backend.edges import EDGE_TYPES
from ..backend.chunks.utils import compute_chunk_id
from ..backend.chunks.hierarchy import get_children_coords


def read_raw_edge_data(imanager, coord) -> Dict:
    edge_dict = _collect_edge_data(imanager, coord)
    edge_dict = postprocess_edge_data(imanager, edge_dict)

    # flag to check if chunk has edges
    # avoid writing to cloud storage if there are no edges
    # unnecessary write operation
    no_edges = True
    chunk_edges = {}
    for edge_type in EDGE_TYPES:
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
    if not no_edges and imanager.cg_meta.data_source.edges:
        put_chunk_edges(imanager.cg_meta.data_source.edges, coord, chunk_edges, 17)
    return chunk_edges


def _get_cont_chunk_coords(imanager, chunk_coord_a, chunk_coord_b):
    """ Computes chunk coordinates that compute data between the named chunks
    :param imanager: IngestionManagaer
    :param chunk_coord_a: np.ndarray
        array of three ints
    :param chunk_coord_b: np.ndarray
        array of three ints
    :return: np.ndarray
    """
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


def _collect_edge_data(imanager, chunk_coord):
    """ Loads edge for single chunk
    :param imanager: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :param aff_dtype: np.dtype
    :param v3_data: bool
    :return: dict of np.ndarrays
    """
    subfolder = "chunked_rg"
    base_path = f"{imanager.cg_meta.data_source.agglomeration}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = defaultdict(list)
    swap = defaultdict(list)
    x, y, z = chunk_coord
    for _x, _y, _z in product([x - 1, x], [y - 1, y], [z - 1, z]):
        if imanager.cg_meta.is_out_of_bounds(np.array([_x, _y, _z])):
            continue
        filename = f"in_chunk_0_{_x}_{_y}_{_z}_{chunk_id}.data"
        filenames[EDGE_TYPES.in_chunk].append(filename)

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            if imanager.cg_meta.is_out_of_bounds(adjacent_chunk_coord):
                continue
            c_chunk_coords = _get_cont_chunk_coords(
                imanager, chunk_coord, adjacent_chunk_coord
            )

            larger_id = np.max([chunk_id, adjacent_chunk_id])
            smaller_id = np.min([chunk_id, adjacent_chunk_id])
            chunk_id_string = f"{smaller_id}_{larger_id}"

            for c_chunk_coord in c_chunk_coords:
                x, y, z = c_chunk_coord
                filename = f"between_chunks_0_{x}_{y}_{z}_{chunk_id_string}.data"
                filenames[EDGE_TYPES.between_chunk].append(filename)
                swap[filename] = larger_id == chunk_id

                # EDGES FROM CUTS OF SVS
                filename = f"fake_0_{x}_{y}_{z}_{chunk_id_string}.data"
                filenames[EDGE_TYPES.cross_chunk].append(filename)
                swap[filename] = larger_id == chunk_id

    edge_data = {}
    read_counter = Counter()
    for k in filenames:
        with cloudvolume.Storage(base_path, n_threads=10) as stor:
            files = stor.get_files(filenames[k])
        data = []
        for file in files:
            if file["error"] or file["content"] is None:
                continue

            edge_dtype = imanager.cg_meta.edge_dtype
            if swap[file["filename"]]:
                this_dtype = [edge_dtype[1], edge_dtype[0]] + edge_dtype[2:]
                content = np.frombuffer(file["content"], dtype=this_dtype)
            else:
                content = np.frombuffer(file["content"], dtype=edge_dtype)

            data.append(content)
            read_counter[k] += 1
        try:
            edge_data[k] = rfn.stack_arrays(data, usemask=False)
        except:
            raise ValueError()

        edge_data_df = pd.DataFrame(edge_data[k])
        edge_data_dfg = (
            edge_data_df.groupby(["sv1", "sv2"]).aggregate(np.sum).reset_index()
        )
        edge_data[k] = edge_data_dfg.to_records()
    return edge_data


def get_active_edges(imanager, coord, edges_d, mapping):
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
    """ Labels edges as within or across segments and extracts isolated ids
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


def read_raw_agglomeration_data(imanager, chunk_coord: np.ndarray):
    """
    Collects agglomeration information & builds connected component mapping
    """
    subfolder = "remap"
    base_path = f"{imanager.cg_meta.data_source.agglomeration}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = []
    for mip_level in range(0, int(imanager.cg_meta.layer_count - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{chunk_id}.data.zst")

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            for mip_level in range(0, int(imanager.cg_meta.layer_count - 1)):
                x, y, z = np.array(adjacent_chunk_coord / 2 ** mip_level, dtype=int)
                filenames.append(
                    f"done_{mip_level}_{x}_{y}_{z}_{adjacent_chunk_id}.data.zst"
                )

    edges_list = _read_agg_files(filenames, base_path)
    G = nx.Graph()
    G.add_edges_from(np.concatenate(edges_list))
    mapping = {}
    components = list(nx.connected_components(G))
    G.clear()
    for i_cc, cc in enumerate(components):
        cc = list(cc)
        mapping.update(dict(zip(cc, [i_cc] * len(cc))))

    if mapping and imanager.cg_meta.data_source.components:
        put_chunk_components(
            imanager.cg_meta.data_source.components, components, chunk_coord
        )
    return mapping


def _read_agg_files(filenames, base_path):
    with cloudvolume.Storage(base_path, n_threads=10) as stor:
        files = stor.get_files(filenames)

    edge_list = []
    for file in files:
        if file["error"] or file["content"] is None:
            continue
        content = zstd.ZstdDecompressor().decompressobj().decompress(file["content"])
        edge_list.append(np.frombuffer(content, dtype=basetypes.NODE_ID).reshape(-1, 2))
    return edge_list
