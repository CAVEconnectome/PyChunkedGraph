import numpy as np
import cloudvolume
import collections
import itertools
import re
import time
import zstandard as zstd
import numpy.lib.recfunctions as rfn
import networkx as nx
import logging
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.ingest import ingestionmanager, ingestion_utils as iu


def ingest_into_chunkedgraph(storage_path, ws_cv_path, cg_table_id,
                             chunk_size=[256, 256, 512],
                             fan_out=2, aff_dtype=np.float32,
                             instance_id=None, project_id=None, n_threads=64):
    """ Creates a chunkedgraph from a Ran Agglomerattion

    :param storage_path: str
        Google cloud bucket path (agglomeration)
        example: gs://ranl-scratch/minnie_test_2
    :param ws_cv_path: str
        Google cloud bucket path (watershed segmentation)
        example: gs://microns-seunglab/minnie_v0/minnie10/ws_minnie_test_2/agg
    :param cg_table_id: str
        chunkedgraph table name
    :param fan_out: int
        fan out of chunked graph (2 == Octree)
    :param aff_dtype: np.dtype
        affinity datatype (np.float32 or np.float64)
    :param instance_id: str
        Google instance id
    :param project_id: str
        Google project id
    :param n_threads: int
        number of threads to use
    :return:
    """
    storage_path = storage_path.strip("/")
    ws_cv_path = ws_cv_path.strip("/")

    cg_mesh_dir = f"{cg_table_id}_meshes"
    chunk_size = np.array(chunk_size, dtype=np.uint64)

    iu.initialize_chunkedgraph(cg_table_id=cg_table_id, ws_cv_path=ws_cv_path,
                               chunk_size=chunk_size,
                               cg_mesh_dir=cg_mesh_dir, fan_out=fan_out,
                               instance_id=instance_id, project_id=project_id)

    im = ingestionmanager.IngestionManager(storage_path=storage_path,
                                           cg_table_id=cg_table_id,
                                           instance_id=instance_id,
                                           project_id=project_id)

    # #TODO: Remove later:
    logging.basicConfig(level=logging.DEBUG)
    im.cg.logger = logging.getLogger(__name__)
    # ------------------------------------------
    create_atomic_chunks(im, aff_dtype=aff_dtype, n_threads=n_threads)
    create_abstract_layers(im, n_threads=n_threads)

    return im


def create_abstract_layers(im, n_threads=1):
    """ Creates abstract of chunkedgraph (> 2)

    :param im: IngestionManager
    :param n_threads: int
        number of threads to use
    :return:
    """

    for layer_id in range(3, int(im.cg.n_layers + 1)):
        create_layer(im, layer_id, n_threads=n_threads)


def create_layer(im, layer_id, n_threads=1):
    """ Creates abstract layer of chunkedgraph

    Abstract layers have to be build in sequence. Abstract layers are all layers
    above the first layer (1). `create_atomic_chunks` creates layer 2 as well.
    Hence, this function is responsible for every creating layers > 2.

    :param im: IngestionManager
    :param layer_id: int
        > 2
    :param n_threads: int
        number of threads to use
    :return:
    """
    assert layer_id > 2

    child_chunk_coords = im.chunk_coords // im.cg.fan_out ** (layer_id - 3)
    child_chunk_coords = child_chunk_coords.astype(np.int)
    child_chunk_coords = np.unique(child_chunk_coords, axis=0)

    parent_chunk_coords = child_chunk_coords // im.cg.fan_out
    parent_chunk_coords = parent_chunk_coords.astype(np.int)
    parent_chunk_coords, inds = np.unique(parent_chunk_coords, axis=0,
                                          return_inverse=True)

    im_info = im.get_serialized_info()
    multi_args = []

    # Randomize chunks
    order = np.arange(len(parent_chunk_coords), dtype=np.int)
    np.random.shuffle(order)

    for i_chunk, idx in enumerate(order):
        multi_args.append([im_info, layer_id, child_chunk_coords[inds == idx],
                           i_chunk, len(order)])

    if n_threads == 1:
        mu.multiprocess_func(
            _create_layer, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_create_layer, multi_args, n_threads=n_threads)


def _create_layer(args):
    """ Multiprocessing helper for create_layer """
    im_info, layer_id, child_chunk_coords, i_chunk, n_chunks = args

    time_start = time.time()

    im = ingestionmanager.IngestionManager(**im_info)
    im.cg.add_layer(layer_id, child_chunk_coords, n_threads=1, verbose=True)

    print(f"\nLayer {layer_id}: {i_chunk} / {n_chunks} -- %.3fs\n" %
          (time.time() - time_start))


def create_atomic_chunks(im, aff_dtype=np.float32, n_threads=1):
    """ Creates all atomic chunks

    :param im: IngestionManager
    :param aff_dtype: np.dtype
        affinity datatype (np.float32 or np.float64)
    :param n_threads: int
        number of threads to use
    :return:
    """

    im_info = im.get_serialized_info()

    multi_args = []

    # Randomize chunk order
    chunk_coords = list(im.chunk_coord_gen)
    # np.random.shuffle(chunk_coords)

    for i_chunk_coord, chunk_coord in enumerate(chunk_coords):
        multi_args.append([im_info, chunk_coord, aff_dtype, i_chunk_coord,
                           len(chunk_coords)])

    if n_threads == 1:
        mu.multiprocess_func(
            _create_atomic_chunk, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(
            _create_atomic_chunk, multi_args, n_threads=n_threads)


def _create_atomic_chunk(args):
    """ Multiprocessing helper for create_atomic_chunks """

    im_info, chunk_coord, aff_dtype, i_chunk, n_chunks = args

    time_start = time.time()

    im = ingestionmanager.IngestionManager(**im_info)
    create_atomic_chunk(im, chunk_coord, aff_dtype=aff_dtype, verbose=True)

    print(f"\nLayer 1/2: {i_chunk} / {n_chunks} -- %.3fs\n" %
          (time.time() - time_start))


def create_atomic_chunk(im, chunk_coord, aff_dtype=np.float32, verbose=True):
    """ Creates single atomic chunk

    :param im: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :param aff_dtype: np.dtype
        np.float64 or np.float32
    :param verbose: bool
    :return:
    """
    chunk_coord = np.array(list(chunk_coord), dtype=np.int)

    edge_dict = collect_edge_data(im, chunk_coord, aff_dtype=aff_dtype)
    mapping = collect_agglomeration_data(im, chunk_coord)
    active_edge_dict, isolated_ids = define_active_edges(edge_dict, mapping)

    edge_ids = {}
    edge_affs = {}
    edge_areas = {}

    for k in edge_dict.keys():
        if k == "cross":
            edge_ids[k] = np.concatenate([edge_dict[k]["sv1"][:, None],
                                          edge_dict[k]["sv2"][:, None]],
                                         axis=1)
            continue

        sv1_conn = edge_dict[k]["sv1"][active_edge_dict[k]]
        sv2_conn = edge_dict[k]["sv2"][active_edge_dict[k]]
        aff_conn = edge_dict[k]["aff"][active_edge_dict[k]]
        area_conn = edge_dict[k]["area"][active_edge_dict[k]]
        edge_ids[f"{k}_connected"] = np.concatenate([sv1_conn[:, None],
                                                     sv2_conn[:, None]],
                                                    axis=1)
        edge_affs[f"{k}_connected"] = aff_conn.astype(np.float32)
        edge_areas[f"{k}_connected"] = area_conn

        sv1_disconn = edge_dict[k]["sv1"][~active_edge_dict[k]]
        sv2_disconn = edge_dict[k]["sv2"][~active_edge_dict[k]]
        aff_disconn = edge_dict[k]["aff"][~active_edge_dict[k]]
        area_disconn = edge_dict[k]["area"][~active_edge_dict[k]]
        edge_ids[f"{k}_disconnected"] = np.concatenate([sv1_disconn[:, None],
                                                        sv2_disconn[:, None]],
                                                       axis=1)
        edge_affs[f"{k}_disconnected"] = aff_disconn.astype(np.float32)
        edge_areas[f"{k}_disconnected"] = area_disconn

    im.cg.add_atomic_edges_in_chunks(edge_ids, edge_affs, edge_areas,
                                     isolated_node_ids=isolated_ids)

    return edge_ids, edge_affs, edge_areas


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


def collect_edge_data(im, chunk_coord, aff_dtype=np.float32):
    """ Loads edge for single chunk

    :param im: IngestionManager
    :param chunk_coord: np.ndarray
        array of three ints
    :param aff_dtype: np.dtype
    :return: dict of np.ndarrays
    """
    subfolder = "chunked_rg"

    base_path = f"{im.storage_path}/{subfolder}/"

    chunk_coord = np.array(chunk_coord)

    chunk_id = im.cg.get_chunk_id(layer=1, x=chunk_coord[0], y=chunk_coord[1],
                                  z=chunk_coord[2])

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
            adjacent_chunk_id = im.cg.get_chunk_id(layer=1,
                                                   x=adjacent_chunk_coord[0],
                                                   y=adjacent_chunk_coord[1],
                                                   z=adjacent_chunk_coord[2])

            if im.is_out_of_bounce(adjacent_chunk_coord):
                continue

            c_chunk_coords = _get_cont_chunk_coords(im, chunk_coord,
                                                    adjacent_chunk_coord)

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

    dtype = [("sv1", np.uint64), ("sv2", np.uint64),
             ("aff", aff_dtype), ("area", np.uint64)]
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
                this_dtype = [dtype[1], dtype[0], dtype[2], dtype[3]]
                content = np.frombuffer(file["content"], dtype=this_dtype)
            else:
                content = np.frombuffer(file["content"], dtype=dtype)

            data.append(content)

            read_counter[k] += 1

        try:
            edge_data[k] = rfn.stack_arrays(data, usemask=False)
        except:
            raise()

    # # TEST
    # with cloudvolume.Storage(base_path, n_threads=10) as stor:
    #     files = list(stor.list_files())
    #
    # true_counter = collections.Counter()
    # for file in files:
    #     if str(chunk_id) in file:
    #         true_counter[file.split("_")[0]] += 1
    #
    # print("Truth", true_counter)
    # print("Reality", read_counter)

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

        content = zstd.ZstdDecompressor().decompress(file["content"])
        edge_list.append(np.frombuffer(content, dtype=np.uint64).reshape(-1, 2))

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

    chunk_id = im.cg.get_chunk_id(layer=1, x=chunk_coord[0], y=chunk_coord[1],
                                  z=chunk_coord[2])

    filenames = []
    for mip_level in range(0, int(im.cg.n_layers - 1)):
        x, y, z = np.array(chunk_coord / 2 ** mip_level, dtype=np.int)
        filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{chunk_id}.data.zst")

    for d in [-1, 1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=np.int)
            diff[dim] = d

            adjacent_chunk_coord = chunk_coord + diff

            adjacent_chunk_id = im.cg.get_chunk_id(layer=1,
                                                   x=adjacent_chunk_coord[0],
                                                   y=adjacent_chunk_coord[1],
                                                   z=adjacent_chunk_coord[2])

            for mip_level in range(0, int(im.cg.n_layers - 1)):
                x, y, z = np.array(adjacent_chunk_coord / 2 ** mip_level, dtype=np.int)
                filenames.append(f"done_{mip_level}_{x}_{y}_{z}_{adjacent_chunk_id}.data.zst")

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
    isolated = []
    for k in edge_dict:
        agg_id_1 = mapping_vec(edge_dict[k]["sv1"])
        agg_id_2 = mapping_vec(edge_dict[k]["sv2"])

        active[k] = agg_id_1 == agg_id_2

        # Set those with two -1 to False
        agg_1_m = agg_id_1 == -1
        agg_2_m = agg_id_2 == -1
        active[k][agg_1_m] = False

        isolated.append(edge_dict[k]["sv1"][agg_1_m])

        if k == "in":
            isolated.append(edge_dict[k]["sv2"][agg_2_m])

    return active, np.unique(np.concatenate(isolated))

