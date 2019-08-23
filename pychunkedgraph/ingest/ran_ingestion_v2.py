"""
Module for ingesting in chunkedgraph format with edges stored outside bigtable
"""

import os
import collections
import time

import click
import pandas as pd
import cloudvolume
import networkx as nx
import numpy as np
import numpy.lib.recfunctions as rfn
import zstandard as zstd
from flask import current_app
from multiwrapper import multiprocessing_utils as mu
from rq import Queue
from redis import Redis

from ..utils.general import redis_job
from . import ingestionmanager, ingestion_utils as iu
from ..backend.initialization.create import add_atomic_edges
from ..backend.definitions.edges import Edges, TYPES as EDGE_TYPES
from ..backend.utils import basetypes
from ..io.edge_storage import put_chunk_edges

REDIS_HOST = os.environ.get("REDIS_SERVICE_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_SERVICE_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
ZSTD_COMPRESSION_LEVEL = 17


def ingest_into_chunkedgraph(
    storage_path,
    ws_cv_path,
    cg_table_id,
    chunk_size=[512, 512, 128],
    use_skip_connections=True,
    fan_out=2,
    size=None,
    instance_id=None,
    project_id=None,
    start_layer=1,
    edge_dir=None,
    n_chunks=None,
    is_new=True
):
    storage_path = storage_path.strip("/")
    ws_cv_path = ws_cv_path.strip("/")

    cg_mesh_dir = f"{cg_table_id}_meshes"
    chunk_size = np.array(chunk_size)

    _, n_layers_agg = iu.initialize_chunkedgraph(
        cg_table_id=cg_table_id,
        ws_cv_path=ws_cv_path,
        chunk_size=chunk_size,
        size=size,
        use_skip_connections=use_skip_connections,
        s_bits_atomic_layer=10,
        cg_mesh_dir=cg_mesh_dir,
        fan_out=fan_out,
        instance_id=instance_id,
        project_id=project_id,
        edge_dir=edge_dir,
        is_new=is_new
    )

    imanager = ingestionmanager.IngestionManager(
        storage_path=storage_path,
        cg_table_id=cg_table_id,
        n_layers=n_layers_agg,
        instance_id=instance_id,
        project_id=project_id,
        data_version=4,
    )

    create_atomic_chunks(imanager, n_chunks)
    return imanager


def queue_parent(im, layer_id, parent_chunk_coord, child_chunk_coords):
    im_info = im.get_serialized_info()
    connection = Redis(host=os.environ['REDIS_SERVICE_HOST'],port=6379,db=0, password='dev')
    test_q = Queue('test', connection=connection)
    test_q.enqueue(
        _create_layer,
        job_timeout="59m",
        args=(im_info, layer_id, child_chunk_coords, parent_chunk_coord))


@redis_job(REDIS_URL, "ingest_channel")
def _create_layer(im_info, layer_id, child_chunk_coords, parent_chunk_coord):
    imanager = ingestionmanager.IngestionManager(**im_info)
    return imanager.cg.add_layer(
        layer_id, child_chunk_coords,
        parent_chunk_coord=parent_chunk_coord)


def create_atomic_chunks(im, n_chunks):
    """ Creates all atomic chunks"""
    chunk_coords = list(im.chunk_coord_gen)
    np.random.shuffle(chunk_coords)

    print(len(chunk_coords))

    for chunk_coord in chunk_coords[:n_chunks]:
        current_app.test_q.enqueue(
            _create_atomic_chunk,
            job_timeout="59m",
            args=(im.get_serialized_info(), chunk_coord),
        )


@redis_job(REDIS_URL, "ingest_channel")
def _create_atomic_chunk(im_info, chunk_coord):
    """ Multiprocessing helper for create_atomic_chunks """
    imanager = ingestionmanager.IngestionManager(**im_info)
    return create_atomic_chunk(imanager, chunk_coord)


def create_atomic_chunk(imanager, chunk_coord):
    """ Creates single atomic chunk"""
    chunk_coord = np.array(list(chunk_coord), dtype=np.int)
    edge_dict = collect_edge_data(imanager, chunk_coord)
    edge_dict = iu.postprocess_edge_data(imanager, edge_dict)
    mapping = collect_agglomeration_data(imanager, chunk_coord)
    _, isolated_ids = define_active_edges(edge_dict, mapping)

    # flag to check if chunk has edges
    # avoid writing to cloud storage if there are no edges
    # unnecessary write operation
    no_edges = True
    chunk_edges = {}
    for edge_type in EDGE_TYPES:
        sv_ids1 = edge_dict[edge_type]["sv1"]
        sv_ids2 = edge_dict[edge_type]["sv2"]

        ones = np.ones(len(sv_ids1))
        affinities = edge_dict[edge_type].get("aff", float("inf") * ones)
        areas = edge_dict[edge_type].get("area", ones)
        chunk_edges[edge_type] = Edges(sv_ids1, sv_ids2, affinities, areas)
        no_edges = no_edges and not sv_ids1.size

    # if not no_edges:
    #     put_chunk_edges(cg.edge_dir, chunk_coord, chunk_edges, ZSTD_COMPRESSION_LEVEL)
    add_atomic_edges(imanager.cg, chunk_coord, chunk_edges, isolated=isolated_ids)

    # to track workers completion
    result = np.concatenate([[2], chunk_coord])
    return result.tobytes()


def _get_cont_chunk_coords(im, chunk_coord_a, chunk_coord_b):
    """ Computes chunk coordinates that compute data between the named chunks

    :param im: IngestionManagaer
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

    if diff[dir_dim] > 0:
        chunk_coord_l = chunk_coord_a
    else:
        chunk_coord_l = chunk_coord_b

    c_chunk_coords = []
    for dx in [-1, 0]:
        for dy in [-1, 0]:
            for dz in [-1, 0]:
                if dz == dy == dx == 0:
                    continue

                c_chunk_coord = chunk_coord_l + np.array([dx, dy, dz])

                if [dx, dy, dz][dir_dim] == 0:
                    continue

                if im.is_out_of_bounce(c_chunk_coord):
                    continue

                c_chunk_coords.append(c_chunk_coord)

    return c_chunk_coords


def collect_edge_data(im, chunk_coord):
    """ Loads edge for single chunk

    :param im: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :param aff_dtype: np.dtype
    :param v3_data: bool
    :return: dict of np.ndarrays
    """
    subfolder = "chunked_rg"

    base_path = f"{im.storage_path}/{subfolder}/"

    chunk_coord = np.array(chunk_coord)

    chunk_id = im.cg.get_chunk_id(
        layer=1, x=chunk_coord[0], y=chunk_coord[1], z=chunk_coord[2]
    )

    filenames = collections.defaultdict(list)
    swap = collections.defaultdict(list)
    for x in [chunk_coord[0] - 1, chunk_coord[0]]:
        for y in [chunk_coord[1] - 1, chunk_coord[1]]:
            for z in [chunk_coord[2] - 1, chunk_coord[2]]:

                if im.is_out_of_bounce(np.array([x, y, z])):
                    continue

                # EDGES WITHIN CHUNKS
                filename = f"in_chunk_0_{x}_{y}_{z}_{chunk_id}.data"
                filenames["in"].append(filename)

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d

            adjacent_chunk_coord = chunk_coord + diff
            adjacent_chunk_id = im.cg.get_chunk_id(
                layer=1,
                x=adjacent_chunk_coord[0],
                y=adjacent_chunk_coord[1],
                z=adjacent_chunk_coord[2],
            )

            if im.is_out_of_bounce(adjacent_chunk_coord):
                continue

            c_chunk_coords = _get_cont_chunk_coords(
                im, chunk_coord, adjacent_chunk_coord
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
        # print(k, len(filenames[k]))

        with cloudvolume.Storage(base_path, n_threads=10) as stor:
            files = stor.get_files(filenames[k])

        data = []
        for file in files:
            if file["content"] is None:
                # print(f"{file['filename']} not created or empty")
                continue

            if file["error"] is not None:
                # print(f"error reading {file['filename']}")
                continue

            if swap[file["filename"]]:
                this_dtype = [im.edge_dtype[1], im.edge_dtype[0]] + im.edge_dtype[2:]
                content = np.frombuffer(file["content"], dtype=this_dtype)
            else:
                content = np.frombuffer(file["content"], dtype=im.edge_dtype)

            data.append(content)

            read_counter[k] += 1

        try:
            edge_data[k] = rfn.stack_arrays(data, usemask=False)
        except:
            raise ()

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
        if file["content"] is None:
            continue

        if file["error"] is not None:
            continue

        content = zstd.ZstdDecompressor().decompressobj().decompress(file["content"])
        edge_list.append(np.frombuffer(content, dtype=basetypes.NODE_ID).reshape(-1, 2))

    return edge_list


def collect_agglomeration_data(im, chunk_coord):
    """ Collects agglomeration information & builds connected component mapping

    :param im: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :return: dictionary
    """
    subfolder = "remap"
    base_path = f"{im.storage_path}/{subfolder}/"

    chunk_coord = np.array(chunk_coord)

    chunk_id = im.cg.get_chunk_id(
        layer=1, x=chunk_coord[0], y=chunk_coord[1], z=chunk_coord[2]
    )

    filenames = []
    for mip_level in range(0, int(im.n_layers - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=np.int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{chunk_id}.data.zst")

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d

            adjacent_chunk_coord = chunk_coord + diff

            adjacent_chunk_id = im.cg.get_chunk_id(
                layer=1,
                x=adjacent_chunk_coord[0],
                y=adjacent_chunk_coord[1],
                z=adjacent_chunk_coord[2],
            )

            for mip_level in range(0, int(im.n_layers - 1)):
                x, y, z = np.array(adjacent_chunk_coord / 2 ** mip_level, dtype=np.int)
                filenames.append(
                    f"done_{mip_level}_{x}_{y}_{z}_{adjacent_chunk_id}.data.zst"
                )

    # print(filenames)
    edge_list = _read_agg_files(filenames, base_path)

    edges = np.concatenate(edge_list)

    G = nx.Graph()
    G.add_edges_from(edges)
    ccs = nx.connected_components(G)

    mapping = {}
    for i_cc, cc in enumerate(ccs):
        cc = list(cc)
        mapping.update(dict(zip(cc, [i_cc] * len(cc))))

    return mapping


def define_active_edges(edge_dict, mapping):
    """ Labels edges as within or across segments and extracts isolated ids

    :param edge_dict: dict of np.ndarrays
    :param mapping: dict
    :return: dict of np.ndarrays, np.ndarray
        bool arrays; True: connected (within same segment)
        isolated node ids
    """

    def _mapping_default(key):
        if key in mapping:
            return mapping[key]
        else:
            return -1

    mapping_vec = np.vectorize(_mapping_default)

    active = {}
    isolated = [[]]
    for k in edge_dict:
        if len(edge_dict[k]["sv1"]) > 0:
            agg_id_1 = mapping_vec(edge_dict[k]["sv1"])
        else:
            assert len(edge_dict[k]["sv2"]) == 0
            active[k] = np.array([], dtype=np.bool)
            continue

        agg_id_2 = mapping_vec(edge_dict[k]["sv2"])

        active[k] = agg_id_1 == agg_id_2

        # Set those with two -1 to False
        agg_1_m = agg_id_1 == -1
        agg_2_m = agg_id_2 == -1
        active[k][agg_1_m] = False

        isolated.append(edge_dict[k]["sv1"][agg_1_m])

        if k == "in":
            isolated.append(edge_dict[k]["sv2"][agg_2_m])

    return active, np.unique(np.concatenate(isolated).astype(basetypes.NODE_ID))
