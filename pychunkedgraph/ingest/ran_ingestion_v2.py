"""
Module for ingesting in chunkedgraph format with edges stored outside bigtable
"""

import collections
import itertools
import json
from typing import Dict, Tuple

import pandas as pd
import cloudvolume
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
import zstandard as zstd
from flask import current_app

from . import ingestionmanager, ingestion_utils as iu
from .initialization.atomic_layer import add_atomic_edges
from .initialization.abstract_layers import add_layer
from ..utils.redis import redis_job, REDIS_URL
from ..io.edges import get_chunk_edges
from ..io.edges import put_chunk_edges
from ..io.agglomeration import get_chunk_agglomeration
from ..io.agglomeration import put_chunk_agglomeration
from ..backend.utils import basetypes
from ..backend.chunkedgraph_utils import compute_bitmasks
from ..backend.chunkedgraph_utils import compute_chunk_id
from ..backend.definitions.edges import Edges, CX_CHUNK
from ..backend.definitions.edges import TYPES as EDGE_TYPES
from ..backend.definitions.config import DataSource
from ..backend.definitions.config import GraphConfig
from ..backend.definitions.config import BigTableConfig


ZSTD_LEVEL = 17
INGEST_CHANNEL = "ingest"
INGEST_QUEUE = "test"

# TODO make sure to generate edges and mappings only once
# add option in data config


def ingest_into_chunkedgraph(
    data_source: DataSource, graph_config: GraphConfig, bigtable_config: BigTableConfig
):
    storage_path = data_source.agglomeration.strip("/")
    ws_cv_path = data_source.watershed.strip("/")
    cg_mesh_dir = f"{graph_config.graph_id}_meshes"
    chunk_size = np.array(graph_config.chunk_size)

    _, n_layers_agg = iu.initialize_chunkedgraph(
        cg_table_id=graph_config.cg_table_id,
        ws_cv_path=ws_cv_path,
        chunk_size=chunk_size,
        use_skip_connections=True,
        s_bits_atomic_layer=10,
        cg_mesh_dir=cg_mesh_dir,
        fan_out=graph_config.fan_out,
        instance_id=bigtable_config.instance_id,
        project_id=bigtable_config.project_id,
        edge_dir=data_source.edges,
        is_new=graph_config.is_new,
    )

    imanager = ingestionmanager.IngestionManager(
        storage_path=storage_path,
        cg_table_id=graph_config.graph_id,
        n_layers=n_layers_agg,
        instance_id=bigtable_config.instance_id,
        project_id=bigtable_config.project_id,
        data_version=4,
        agglomeration_dir=data_source.components,
        use_raw_edge_data=(data_source.edges == None),
        use_raw_agglomeration_data=(data_source.components == None),
    )
    return imanager


@redis_job(REDIS_URL, INGEST_CHANNEL)
def create_parent_chunk(im_info, layer, child_chunk_coords):
    imanager = ingestionmanager.IngestionManager(**im_info)
    return add_layer(imanager.cg, layer, child_chunk_coords)


def enqueue_atomic_tasks(imanager):
    """ Creates all atomic chunks"""
    chunk_coords = list(imanager.chunk_coord_gen)
    np.random.shuffle(chunk_coords)
    print(f"Chunk count: {len(chunk_coords)}")
    for chunk_coord in chunk_coords:
        current_app.test_q.enqueue(
            _create_atomic_chunk,
            job_timeout="59m",
            args=(imanager.get_serialized_info(), chunk_coord),
        )
    print(f"Queued jobs: {len(current_app.test_q)}")


@redis_job(REDIS_URL, INGEST_CHANNEL)
def _create_atomic_chunk(im_info, chunk_coord):
    """ helper for enqueue_atomic_tasks """
    imanager = ingestionmanager.IngestionManager(**im_info)
    return create_atomic_chunk(imanager, chunk_coord)


def create_atomic_chunk(imanager, coord):
    """ Creates single atomic chunk"""
    coord = np.array(list(coord), dtype=np.int)
    chunk_edges_all, mapping = _get_chunk_data(imanager, coord)
    chunk_edges_active, isolated_ids = _get_active_edges(
        imanager, coord, chunk_edges_all, mapping
    )
    add_atomic_edges(imanager.cg, coord, chunk_edges_active, isolated=isolated_ids)

    n_supervoxels = len(isolated_ids)
    n_edges = 0
    for edge_type in EDGE_TYPES:
        edges = chunk_edges_all[edge_type]
        n_edges += len(edges)
        n_supervoxels += len(np.unique(edges.ravel()))
    return ",".join(
        map(str, [f"{2}_{'_'.join(map(str, coord))}", n_supervoxels, n_edges])
    )


def _get_chunk_data(imanager, coord) -> Tuple[Dict, Dict]:
    """
    Helper to read either raw data or processed data
    If reading from raw data, save it as processed data
    """
    chunk_edges = (
        _read_raw_edge_data(imanager, coord)
        if imanager.use_raw_edge_data
        else get_chunk_edges(imanager.cg.cv_edges_path, [coord])
    )
    mapping = (
        _read_raw_agglomeration_data(imanager, coord)
        if imanager.use_raw_agglomeration_data
        else get_chunk_agglomeration(imanager.agglomeration_dir, coord)
    )
    return chunk_edges, mapping


def _read_raw_edge_data(imanager, coord) -> Dict:
    edge_dict = _collect_edge_data(imanager, coord)
    edge_dict = iu.postprocess_edge_data(imanager, edge_dict)

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
        if not edge_type == CX_CHUNK:
            affinities = edge_dict[edge_type]["aff"]
            areas = edge_dict[edge_type]["area"]

        chunk_edges[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
        no_edges = no_edges and not sv_ids1.size
    if no_edges:
        return None
    put_chunk_edges(imanager.cg.cv_edges_path, coord, chunk_edges, ZSTD_LEVEL)
    return chunk_edges


def _get_active_edges(imanager, coord, edges_d, mapping):
    active_edges_flag_d, isolated_ids = _define_active_edges(edges_d, mapping)
    chunk_edges_active = {}
    for edge_type in EDGE_TYPES:
        edges = edges_d[edge_type]
        active = active_edges_flag_d[edge_type]

        sv_ids1 = edges.node_ids1[active]
        sv_ids2 = edges.node_ids2[active]
        affinities = edges.affinities[active]
        areas = edges.areas[active]
        chunk_edges_active[edge_type] = Edges(
            sv_ids1, sv_ids2, affinities=affinities, areas=areas
        )
    return chunk_edges_active, isolated_ids


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
    for dx, dy, dz in itertools.product([0, -1], [0, -1], [0, -1]):
        if dz == dy == dx == 0:
            continue
        if [dx, dy, dz][dir_dim] == 0:
            continue

        c_chunk_coord = chunk_coord_l + np.array([dx, dy, dz])
        if imanager.is_out_of_bounds(c_chunk_coord):
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
    base_path = f"{imanager.storage_path}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = collections.defaultdict(list)
    swap = collections.defaultdict(list)
    x, y, z = chunk_coord
    for _x, _y, _z in itertools.product([x - 1, x], [y - 1, y], [z - 1, z]):
        if imanager.is_out_of_bounds(np.array([_x, _y, _z])):
            continue
        # EDGES WITHIN CHUNKS
        filename = f"in_chunk_0_{_x}_{_y}_{_z}_{chunk_id}.data"
        filenames["in"].append(filename)

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            if imanager.is_out_of_bounds(adjacent_chunk_coord):
                continue
            c_chunk_coords = _get_cont_chunk_coords(
                imanager, chunk_coord, adjacent_chunk_coord
            )

            larger_id = np.max([chunk_id, adjacent_chunk_id])
            smaller_id = np.min([chunk_id, adjacent_chunk_id])
            chunk_id_string = f"{smaller_id}_{larger_id}"

            for c_chunk_coord in c_chunk_coords:
                x, y, z = c_chunk_coord
                # EDGES BETWEEN CHUNKS
                filename = f"between_chunks_0_{x}_{y}_{z}_{chunk_id_string}.data"
                filenames["between"].append(filename)
                swap[filename] = larger_id == chunk_id

                # EDGES FROM CUTS OF SVS
                filename = f"fake_0_{x}_{y}_{z}_{chunk_id_string}.data"
                filenames["cross"].append(filename)
                swap[filename] = larger_id == chunk_id

    edge_data = {}
    read_counter = collections.Counter()
    for k in filenames:
        with cloudvolume.Storage(base_path, n_threads=10) as stor:
            files = stor.get_files(filenames[k])
        data = []
        for file in files:
            if file["error"] or file["content"] is None:
                continue

            if swap[file["filename"]]:
                this_dtype = [
                    imanager.edge_dtype[1],
                    imanager.edge_dtype[0],
                ] + imanager.edge_dtype[2:]
                content = np.frombuffer(file["content"], dtype=this_dtype)
            else:
                content = np.frombuffer(file["content"], dtype=imanager.edge_dtype)

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


def _read_raw_agglomeration_data(imanager, chunk_coord):
    """ Collects agglomeration information & builds connected component mapping
    :param imanager: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :return: dictionary
    """
    subfolder = "remap"
    base_path = f"{imanager.storage_path}/{subfolder}/"
    chunk_coord = np.array(chunk_coord)
    x, y, z = chunk_coord
    chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

    filenames = []
    for mip_level in range(0, int(imanager.n_layers - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=np.int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{chunk_id}.data.zst")

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d
            adjacent_chunk_coord = chunk_coord + diff
            x, y, z = adjacent_chunk_coord
            adjacent_chunk_id = compute_chunk_id(layer=1, x=x, y=y, z=z)

            for mip_level in range(0, int(imanager.n_layers - 1)):
                x, y, z = np.array(adjacent_chunk_coord / 2 ** mip_level, dtype=np.int)
                filenames.append(
                    f"done_{mip_level}_{x}_{y}_{z}_{adjacent_chunk_id}.data.zst"
                )

    edges_list = _read_agg_files(filenames, base_path)
    G = nx.Graph()
    G.add_edges_from(np.concatenate(edges_list))
    mapping = {}
    for i_cc, cc in enumerate(nx.connected_components(G)):
        cc = list(cc)
        mapping.update(dict(zip(cc, [i_cc] * len(cc))))

    if mapping:
        put_chunk_agglomeration(imanager.agglomeration_dir, mapping, chunk_coord)
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
