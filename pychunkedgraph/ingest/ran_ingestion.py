"""
Module for ingesting in chunkedgraph format with edges stored outside bigtable
"""

import time
import json
from collections import defaultdict
from collections import Counter
from itertools import product
from typing import Dict
from typing import Tuple
from typing import Sequence

import pandas as pd
import cloudvolume
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
import zstandard as zstd

from .ingestion_utils import postprocess_edge_data
from .manager import IngestionManager

from ..backend import ChunkedGraphMeta
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.components import get_chunk_components
from ..io.components import put_chunk_components
from ..backend.utils import basetypes
from ..backend.chunkedgraph_utils import compute_chunk_id
from ..backend.definitions.edges import Edges
from ..backend.definitions.edges import EDGE_TYPES

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def _get_children_coords(
    cg_meta: ChunkedGraphMeta, layer: int, chunk_coords
) -> np.ndarray:
    chunk_coords = np.array(chunk_coords, dtype=int)
    children_layer = layer - 1
    layer_boundaries = cg_meta.layer_chunk_bounds[children_layer]
    children_coords = []

    for dcoord in product(*[range(cg_meta.graph_config.fanout)] * 3):
        dcoord = np.array(dcoord, dtype=int)
        child_coords = chunk_coords * cg_meta.graph_config.fanout + dcoord
        check_bounds = np.less(child_coords, layer_boundaries)
        if np.all(check_bounds):
            children_coords.append(child_coords)
    return children_coords


def _post_task_completion(imanager: IngestionManager, layer: int, coords: np.ndarray):
    chunk_str = "_".join(map(str, coords))
    # remove from queued hash and put in completed hash
    imanager.redis.hdel(f"{layer}q", chunk_str)
    imanager.redis.hset(f"{layer}c", chunk_str, "")

    parent_layer = layer + 1
    if parent_layer > imanager.chunkedgraph_meta.layer_count:
        return

    parent_coords = (
        np.array(coords, int) // imanager.chunkedgraph_meta.graph_config.fanout
    )
    parent_chunk_str = "_".join(map(str, parent_coords))
    if not imanager.redis.hget(parent_layer, parent_chunk_str):
        children_count = len(
            _get_children_coords(
                imanager.chunkedgraph_meta, parent_layer, parent_coords
            )
        )
        imanager.redis.hset(parent_layer, parent_chunk_str, children_count)
    imanager.redis.hincrby(parent_layer, parent_chunk_str, -1)
    children_left = int(
        imanager.redis.hget(parent_layer, parent_chunk_str).decode("utf-8")
    )

    if children_left == 0:
        parents_queue = imanager.get_task_queue(imanager.config.parents_q_name)
        parents_queue.enqueue(
            _create_parent_chunk,
            job_id=chunk_id_str(parent_layer, parent_coords),
            job_timeout="59m",
            result_ttl=0,
            args=(
                imanager.get_serialized_info(),
                parent_layer,
                parent_coords,
                _get_children_coords(
                    imanager.chunkedgraph_meta, parent_layer, parent_coords
                ),
            ),
        )
        imanager.redis.hdel(parent_layer, parent_chunk_str)
        imanager.redis.hset(f"{parent_layer}q", parent_chunk_str, "")


def _create_parent_chunk(im_info, layer, parent_coords, child_chunk_coords):
    imanager = IngestionManager(**im_info)
    add_layer(imanager.cg, layer, parent_coords, child_chunk_coords)
    _post_task_completion(imanager, layer, parent_coords)


def enqueue_atomic_tasks(
    imanager: IngestionManager, batch_size: int = 50000, interval: float = 300.0
):
    atomic_chunk_bounds = imanager.chunkedgraph_meta.layer_chunk_bounds[2]
    chunk_coords = list(product(*[range(r) for r in atomic_chunk_bounds]))
    np.random.shuffle(chunk_coords)

    for chunk_coord in chunk_coords:
        atomic_queue = imanager.get_task_queue(imanager.config.atomic_q_name)
        # for optimal use of redis memory wait if queue limit is reached
        if len(atomic_queue) > imanager.config.atomic_q_limit:
            print(f"Sleeping {imanager.config.atomic_q_interval}s...")
            time.sleep(imanager.config.atomic_q_interval)
        atomic_queue.enqueue(
            _create_atomic_chunk,
            job_id=chunk_id_str(2, chunk_coord),
            job_timeout="59m",
            result_ttl=0,
            args=(imanager.get_serialized_info(), chunk_coord),
        )


def _create_atomic_chunk(im_info, coord):
    """ Creates single atomic chunk"""
    imanager = IngestionManager(**im_info)
    coord = np.array(list(coord), dtype=np.int)
    chunk_edges_all, mapping = _get_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = _get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)
    _post_task_completion(imanager, 2, coord)


def _get_chunk_data(imanager, coord) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        _read_raw_edge_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_edges
        else get_chunk_edges(imanager.chunkedgraph_meta.data_source.edges, [coord])
    )
    mapping = (
        _read_raw_agglomeration_data(imanager, coord)
        if imanager.chunkedgraph_meta.data_source.use_raw_components
        else get_chunk_components(
            imanager.chunkedgraph_meta.data_source.components, coord
        )
    )
    return chunk_edges, mapping


def _read_raw_edge_data(imanager, coord) -> Dict:
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
    if no_edges:
        return chunk_edges
    put_chunk_edges(
        imanager.chunkedgraph_meta.data_source.edges, coord, chunk_edges, 17
    )
    return chunk_edges


def _get_active_edges(imanager, coord, edges_d, mapping):
    active_edges_flag_d, isolated_ids = _define_active_edges(edges_d, mapping)
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
        if imanager.chunkedgraph_meta.is_out_of_bounds(c_chunk_coord):
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
    base_path = f"{imanager.chunkedgraph_meta.data_source.agglomeration}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = defaultdict(list)
    swap = defaultdict(list)
    x, y, z = chunk_coord
    for _x, _y, _z in product([x - 1, x], [y - 1, y], [z - 1, z]):
        if imanager.chunkedgraph_meta.is_out_of_bounds(np.array([_x, _y, _z])):
            continue
        filename = f"in_chunk_0_{_x}_{_y}_{_z}_{chunk_id}.data"
        filenames["in"].append(filename)

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            if imanager.chunkedgraph_meta.is_out_of_bounds(adjacent_chunk_coord):
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
                filenames["between"].append(filename)
                swap[filename] = larger_id == chunk_id

                # EDGES FROM CUTS OF SVS
                filename = f"fake_0_{x}_{y}_{z}_{chunk_id_string}.data"
                filenames["cross"].append(filename)
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

            edge_dtype = imanager.chunkedgraph_meta.edge_dtype
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


def _read_raw_agglomeration_data(imanager, chunk_coord: np.ndarray):
    """
    Collects agglomeration information & builds connected component mapping
    """
    subfolder = "remap"
    base_path = f"{imanager.chunkedgraph_meta.data_source.agglomeration}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = []
    for mip_level in range(0, int(imanager.chunkedgraph_meta.layer_count - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=np.int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{chunk_id}.data.zst")

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            for mip_level in range(0, int(imanager.chunkedgraph_meta.layer_count - 1)):
                x, y, z = np.array(adjacent_chunk_coord / 2 ** mip_level, dtype=np.int)
                filenames.append(
                    f"done_{mip_level}_{x}_{y}_{z}_{adjacent_chunk_id}.data.zst"
                )

    edges_list = _read_agg_files(filenames, base_path)
    G = nx.Graph()
    G.add_edges_from(np.concatenate(edges_list))
    mapping = {}
    components = list(nx.connected_components(G))
    for i_cc, cc in enumerate(components):
        cc = list(cc)
        mapping.update(dict(zip(cc, [i_cc] * len(cc))))

    if mapping:
        put_chunk_components(
            imanager.chunkedgraph_meta.data_source.components, components, chunk_coord
        )
    return mapping


def _define_active_edges(edge_dict, mapping):
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
            active[k] = np.array([], dtype=np.bool)
            continue

        agg_id_2 = mapping_vec(edge_dict[k].node_ids2)
        active[k] = agg_id_1 == agg_id_2
        # Set those with two -1 to False
        agg_1_m = agg_id_1 == -1
        agg_2_m = agg_id_2 == -1
        active[k][agg_1_m] = False

        isolated.append(edge_dict[k].node_ids1[agg_1_m])
        if k == "in":
            isolated.append(edge_dict[k].node_ids2[agg_2_m])
    return active, np.unique(np.concatenate(isolated).astype(basetypes.NODE_ID))
