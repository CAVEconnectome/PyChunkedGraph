import glob
import numpy as np
import os
import re
import time
import itertools
import random

from cloudvolume import storage

# from chunkedgraph import ChunkedGraph
import pychunkedgraph.backend.chunkedgraph_utils
from pychunkedgraph.backend import chunkedgraph
from multiwrapper import multiprocessing_utils as mu
from pychunkedgraph.creator import creator_utils


def download_and_store_cv_files(dataset_name="basil",
                                n_threads=10, olduint32=False):
    """ Downloads files from google cloud using cloud-volume

    :param dataset_name: str
    :param n_threads: int
    :param olduint32: bool
    """
    if "basil" == dataset_name:
        cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
    elif "pinky40" == dataset_name:
        cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
    elif "pinky100" == dataset_name:
        cv_url = "gs://nkem/pinky100_v0/region_graph/"
    else:
        raise Exception("Could not identify region graph ressource")

    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = creator_utils.dir_from_layer_name(
            creator_utils.layer_name_from_cv_url(cv_st.layer_path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_paths = list(cv_st.list_files())

    file_chunks = np.array_split(file_paths, n_threads * 3)
    multi_args = []
    for i_file_chunk, file_chunk in enumerate(file_chunks):
        multi_args.append([i_file_chunk, cv_url, file_chunk, olduint32])

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(_download_and_store_cv_files_thread,
                             multi_args, n_threads=n_threads,
                             verbose=True, debug=n_threads==1)
    else:
        mu.multisubprocess_func(_download_and_store_cv_files_thread,
                                multi_args, n_threads=n_threads)


def _download_and_store_cv_files_thread(args):
    """ Helper thread to download files from google cloud """
    chunk_id, cv_url, file_paths, olduint32 = args

    # Reset connection pool to make cloud-volume compatible with parallelizing
    storage.reset_connection_pools()

    n_file_paths = len(file_paths)
    time_start = time.time()
    with storage.SimpleStorage(cv_url) as cv_st:
        for i_fp, fp in enumerate(file_paths):
            if i_fp % 100 == 1:
                dt = time.time() - time_start
                eta = dt / i_fp * n_file_paths - dt
                print("%d: %d / %d - dt: %.3fs - eta: %.3fs" % (
                chunk_id, i_fp, n_file_paths, dt, eta))

            creator_utils.download_and_store_edge_file(cv_st, fp)


def check_stored_cv_files(dataset_name="basil"):
    """ Tests if all files were downloaded

    :param dataset_name: str
    """
    if "basil" == dataset_name:
        cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
    elif "pinky40" == dataset_name:
        cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
    elif "pinky100" == dataset_name:
        cv_url = "gs://nkem/pinky100_v0/region_graph/"
    else:
        raise Exception("Could not identify region graph ressource")

    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = creator_utils.dir_from_layer_name(
            creator_utils.layer_name_from_cv_url(cv_st.layer_path))

        file_paths = list(cv_st.list_files())

    c = 0
    n_file_paths = len(file_paths)
    time_start = time.time()
    for i_fp, fp in enumerate(file_paths):
        if i_fp % 1000 == 1:
            dt = time.time() - time_start
            eta = dt / i_fp * n_file_paths - dt
            print("%d / %d - dt: %.3fs - eta: %.3fs" % (
            i_fp, n_file_paths, dt, eta))

        if not os.path.exists(dir_path + fp[:-4] + ".h5"):
            print(dir_path + fp[:-4] + ".h5")
            c += 1

    print("%d files were missing" % c)


def _sort_arrays(coords, paths):
    sorting = np.lexsort((coords[..., 2], coords[..., 1], coords[..., 0]))
    return coords[sorting], paths[sorting]

def create_chunked_graph(table_id=None, cv_url=None, ws_url=None, fan_out=2,
                         bbox=None, chunk_size=(512, 512, 128), verbose=False,
                         n_threads=1):
    """ Creates chunked graph from downloaded files

    :param table_id: str
    :param cv_url: str
    :param ws_url: str
    :param fan_out: int
    :param bbox: [[x_, y_, z_], [_x, _y, _z]]
    :param chunk_size: tuple
    :param verbose: bool
    :param n_threads: int
    """
    if cv_url is None or ws_url is None:
        if "basil" in table_id:
            cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
            ws_url = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
        elif "pinky40" in table_id:
            cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
            ws_url = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
        elif "pinky100" in table_id:
            cv_url = "gs://nkem/pinky100_v0/region_graph/"
            ws_url = "gs://neuroglancer/nkem/pinky100_v0/ws/lost_no-random/bbox1_0/"
        else:
            raise Exception("Could not identify region graph ressource")

    times = []
    time_start = time.time()

    chunk_size = np.array(list(chunk_size))

    file_paths = np.sort(glob.glob(creator_utils.dir_from_layer_name(
        creator_utils.layer_name_from_cv_url(cv_url)) + "/*"))

    file_path_blocks = np.array_split(file_paths, n_threads * 3)

    multi_args = []
    for fp_block in file_path_blocks:
        multi_args.append([fp_block, table_id, chunk_size, bbox])

    if n_threads == 1:
        results = mu.multiprocess_func(
            _preprocess_chunkedgraph_data_thread, multi_args,
            n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(
            _preprocess_chunkedgraph_data_thread, multi_args,
            n_threads=n_threads)

    in_chunk_connected_paths = np.array([])
    in_chunk_connected_ids = np.array([], dtype=np.uint64).reshape(-1, 3)
    in_chunk_disconnected_paths = np.array([])
    in_chunk_disconnected_ids = np.array([], dtype=np.uint64).reshape(-1, 3)
    between_chunk_paths = np.array([])
    between_chunk_ids = np.array([], dtype=np.uint64).reshape(-1, 2, 3)
    isolated_paths = np.array([])
    isolated_ids = np.array([], dtype=np.uint64).reshape(-1, 3)

    for result in results:
        in_chunk_connected_paths = np.concatenate([in_chunk_connected_paths, result[0]])
        in_chunk_connected_ids = np.concatenate([in_chunk_connected_ids, result[1]])
        in_chunk_disconnected_paths = np.concatenate([in_chunk_disconnected_paths, result[2]])
        in_chunk_disconnected_ids = np.concatenate([in_chunk_disconnected_ids, result[3]])
        between_chunk_paths = np.concatenate([between_chunk_paths, result[4]])
        between_chunk_ids = np.concatenate([between_chunk_ids, result[5]])
        isolated_paths = np.concatenate([isolated_paths, result[6]])
        isolated_ids = np.concatenate([isolated_ids, result[7]])

    assert len(in_chunk_connected_ids) == len(in_chunk_connected_paths) == \
           len(in_chunk_disconnected_ids) == len(in_chunk_disconnected_paths) == \
           len(isolated_ids) == len(isolated_paths)

    in_chunk_connected_ids, in_chunk_connected_paths = \
        _sort_arrays(in_chunk_connected_ids, in_chunk_connected_paths)

    in_chunk_disconnected_ids, in_chunk_disconnected_paths = \
        _sort_arrays(in_chunk_disconnected_ids, in_chunk_disconnected_paths)

    isolated_ids, isolated_paths = \
        _sort_arrays(isolated_ids, isolated_paths)

    times.append(["Preprocessing", time.time() - time_start])

    print("Preprocessing took %.3fs = %.2fh" % (times[-1][1], times[-1][1]/3600))

    time_start = time.time()

    multi_args = []

    in_chunk_id_blocks = np.array_split(in_chunk_connected_ids, max(1, n_threads))
    cumsum = 0

    for in_chunk_id_block in in_chunk_id_blocks:
        multi_args.append([between_chunk_ids, between_chunk_paths,
                           in_chunk_id_block, cumsum])
        cumsum += len(in_chunk_id_block)

    # Run parallelizing
    if n_threads == 1:
        results = mu.multiprocess_func(
            _between_chunk_masks_thread, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(
            _between_chunk_masks_thread, multi_args, n_threads=n_threads)

    times.append(["Data sorting", time.time() - time_start])

    print("Data sorting took %.3fs = %.2fh" % (times[-1][1], times[-1][1]/3600))

    time_start = time.time()

    n_layers = int(np.ceil(pychunkedgraph.backend.chunkedgraph_utils.log_n(np.max(in_chunk_connected_ids) + 1, fan_out))) + 2

    print("N layers: %d" % n_layers)

    cg = chunkedgraph.ChunkedGraph(table_id=table_id, n_layers=np.uint64(n_layers),
                                   fan_out=np.uint64(fan_out),
                                   chunk_size=np.array(chunk_size, dtype=np.uint64),
                                   cv_path=ws_url, is_new=True)

    # Fill lowest layer and create first abstraction layer
    # Create arguments for parallelizing

    multi_args = []
    for result in results:
        offset, between_chunk_paths_out_masked, between_chunk_paths_in_masked = result

        for i_chunk in range(len(between_chunk_paths_out_masked)):
            multi_args.append([table_id,
                               in_chunk_connected_paths[offset + i_chunk],
                               in_chunk_disconnected_paths[offset + i_chunk],
                               isolated_paths[offset + i_chunk],
                               between_chunk_paths_in_masked[i_chunk],
                               between_chunk_paths_out_masked[i_chunk],
                               verbose])

    random.shuffle(multi_args)

    print("%d jobs for creating layer 1 + 2" % len(multi_args))

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads)

    times.append(["Layers 1 + 2", time.time() - time_start])

    # Fill higher abstraction layers
    child_chunk_ids = in_chunk_connected_ids.copy()
    for layer_id in range(3, n_layers + 1):

        time_start = time.time()

        print("\n\n\n --- LAYER %d --- \n\n\n" % layer_id)

        parent_chunk_ids = child_chunk_ids // cg.fan_out
        parent_chunk_ids = parent_chunk_ids.astype(int)

        u_pcids, inds = np.unique(parent_chunk_ids,
                                  axis=0, return_inverse=True)

        if len(u_pcids) > n_threads:
            n_threads_per_process = 1
        else:
            n_threads_per_process = int(np.ceil(n_threads / len(u_pcids)))

        multi_args = []
        for ind in range(len(u_pcids)):
            multi_args.append([table_id, layer_id,
                               child_chunk_ids[inds == ind].astype(int),
                               n_threads_per_process])

        child_chunk_ids = u_pcids

        # Run parallelizing
        if n_threads == 1:
            mu.multiprocess_func(
                _add_layer_thread, multi_args, n_threads=n_threads,
                verbose=True,
                debug=n_threads == 1)
        else:
            mu.multisubprocess_func(
                _add_layer_thread, multi_args, n_threads=n_threads,
                suffix=str(layer_id))

        times.append(["Layer %d" % layer_id, time.time() - time_start])

    for time_entry in times:
        print("%s: %.2fs = %.2fmin = %.2fh" % (time_entry[0], time_entry[1],
                                               time_entry[1] / 60,
                                               time_entry[1] / 3600))


def _preprocess_chunkedgraph_data_thread(args):
    """ Reads downloaded files and sorts them in _in_ and _between_ chunks """

    file_paths, table_id, chunk_size, bbox = args

    if bbox is None:
        bbox = [[0, 0, 0], [np.inf, np.inf, np.inf]]

    bbox = np.array(bbox)

    in_chunk_connected_paths = np.array([])
    in_chunk_connected_ids = np.array([], dtype=np.uint64).reshape(-1, 3)
    in_chunk_disconnected_paths = np.array([])
    in_chunk_disconnected_ids = np.array([], dtype=np.uint64).reshape(-1, 3)
    between_chunk_paths = np.array([])
    between_chunk_ids = np.array([], dtype=np.uint64).reshape(-1, 2, 3)
    isolated_paths = np.array([])
    isolated_ids = np.array([], dtype=np.uint64).reshape(-1, 3)

    # Read file paths - gather chunk ids and in / out properties
    for i_fp, fp in enumerate(file_paths):
        file_name = os.path.basename(fp).split(".")[0]

        # Read coordinates from file path
        x1, x2, y1, y2, z1, z2 = np.array(re.findall("[\d]+", file_name), dtype=int)[:6]

        if np.any((bbox[0] - np.array([x2, y2, z2])) >= 0) or \
                np.any((bbox[1] - np.array([x1, y1, z1])) <= 0):
            continue

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        d = np.array([dx, dy, dz])
        c = np.array([x1, y1, z1])

        # if there is a 2 in d then the file contains edges that cross chunks
        gap = 2

        if gap in d:
            s_c = np.where(d == gap)[0]
            chunk_coord = c.copy()

            chunk1_id = np.array(chunk_coord / chunk_size, dtype=int)
            chunk_coord[s_c] += chunk_size[s_c]
            chunk2_id = np.array(chunk_coord / chunk_size, dtype=int)

            between_chunk_ids = np.concatenate([between_chunk_ids,
                                                np.array([chunk1_id, chunk2_id])[None]])
            between_chunk_paths = np.concatenate([between_chunk_paths, [fp]])
        else:
            chunk_coord = np.array(c / chunk_size, dtype=int)

            if "disconnected" in file_name:
                in_chunk_disconnected_ids = np.concatenate([in_chunk_disconnected_ids, chunk_coord[None]])
                in_chunk_disconnected_paths = np.concatenate([in_chunk_disconnected_paths, [fp]])
            elif "isolated" in file_name:
                isolated_ids = np.concatenate([isolated_ids, chunk_coord[None]])
                isolated_paths = np.concatenate([isolated_paths, [fp]])
            else:
                in_chunk_connected_ids = np.concatenate([in_chunk_connected_ids, chunk_coord[None]])
                in_chunk_connected_paths = np.concatenate([in_chunk_connected_paths, [fp]])

    return in_chunk_connected_paths, in_chunk_connected_ids, \
           in_chunk_disconnected_paths, in_chunk_disconnected_ids, \
           between_chunk_paths, between_chunk_ids, \
           isolated_paths, isolated_ids


def _between_chunk_masks_thread(args):
    """"""
    between_chunk_ids, between_chunk_paths, in_chunk_id_block, offset = args

    between_chunk_paths_out_masked = []
    between_chunk_paths_in_masked = []

    for i_in_chunk_id, in_chunk_id in enumerate(in_chunk_id_block):
        out_paths_mask = np.sum(np.abs(between_chunk_ids[:, 0] - in_chunk_id), axis=1) == 0
        in_paths_masks = np.sum(np.abs(between_chunk_ids[:, 1] - in_chunk_id), axis=1) == 0

        between_chunk_paths_out_masked.append(between_chunk_paths[out_paths_mask])
        between_chunk_paths_in_masked.append(between_chunk_paths[in_paths_masks])

    return offset, between_chunk_paths_out_masked, between_chunk_paths_in_masked


def _create_atomic_layer_thread(args):
    """ Fills lowest layer and create first abstraction layer """
    # Load args
    table_id, chunk_connected_path, chunk_disconnected_path, isolated_path,\
        in_paths, out_paths, verbose = args

    # Load edge information
    edge_ids = {"in_connected": np.array([], dtype=np.uint64).reshape(0, 2),
                "in_disconnected": np.array([], dtype=np.uint64).reshape(0, 2),
                "cross": np.array([], dtype=np.uint64).reshape(0, 2),
                "between_connected": np.array([], dtype=np.uint64).reshape(0, 2),
                "between_disconnected": np.array([], dtype=np.uint64).reshape(0, 2)}
    edge_affs = {"in_connected": np.array([], dtype=np.float32),
                 "in_disconnected": np.array([], dtype=np.float32),
                 "between_connected": np.array([], dtype=np.float32),
                 "between_disconnected": np.array([], dtype=np.float32)}
    edge_areas = {"in_connected": np.array([], dtype=np.float32),
                  "in_disconnected": np.array([], dtype=np.float32),
                  "between_connected": np.array([], dtype=np.float32),
                  "between_disconnected": np.array([], dtype=np.float32)}

    in_connected_dict = creator_utils.read_edge_file_h5(chunk_connected_path)
    in_disconnected_dict = creator_utils.read_edge_file_h5(chunk_disconnected_path)

    edge_ids["in_connected"] = in_connected_dict["edge_ids"]
    edge_affs["in_connected"] = in_connected_dict["edge_affs"]
    edge_areas["in_connected"] = in_connected_dict["edge_areas"]

    edge_ids["in_disconnected"] = in_disconnected_dict["edge_ids"]
    edge_affs["in_disconnected"] = in_disconnected_dict["edge_affs"]
    edge_areas["in_disconnected"] = in_disconnected_dict["edge_areas"]

    if os.path.exists(isolated_path):
        isolated_ids = creator_utils.read_edge_file_h5(isolated_path)["node_ids"]
    else:
        isolated_ids = np.array([], dtype=np.uint64)

    for fp in in_paths:
        edge_dict = creator_utils.read_edge_file_h5(fp)

        # Cross edges are always ordered to point OUT of the chunk
        if "unbreakable" in fp:
            edge_ids["cross"] = np.concatenate([edge_ids["cross"], edge_dict["edge_ids"][:, [1, 0]]])
        elif "disconnected" in fp:
            edge_ids["between_disconnected"] = np.concatenate([edge_ids["between_disconnected"], edge_dict["edge_ids"][:, [1, 0]]])
            edge_affs["between_disconnected"] = np.concatenate([edge_affs["between_disconnected"], edge_dict["edge_affs"]])
            edge_areas["between_disconnected"] = np.concatenate([edge_areas["between_disconnected"], edge_dict["edge_areas"]])
        else:
            # connected
            edge_ids["between_connected"] = np.concatenate([edge_ids["between_connected"], edge_dict["edge_ids"][:, [1, 0]]])
            edge_affs["between_connected"] = np.concatenate([edge_affs["between_connected"], edge_dict["edge_affs"]])
            edge_areas["between_connected"] = np.concatenate([edge_areas["between_connected"], edge_dict["edge_areas"]])

    for fp in out_paths:
        edge_dict = creator_utils.read_edge_file_h5(fp)

        if "unbreakable" in fp:
            edge_ids["cross"] = np.concatenate([edge_ids["cross"], edge_dict["edge_ids"]])
        elif "disconnected" in fp:
            edge_ids["between_disconnected"] = np.concatenate([edge_ids["between_disconnected"], edge_dict["edge_ids"]])
            edge_affs["between_disconnected"] = np.concatenate([edge_affs["between_disconnected"], edge_dict["edge_affs"]])
            edge_areas["between_disconnected"] = np.concatenate([edge_areas["between_disconnected"], edge_dict["edge_areas"]])
        else:
            # connected
            edge_ids["between_connected"] = np.concatenate([edge_ids["between_connected"], edge_dict["edge_ids"]])
            edge_affs["between_connected"] = np.concatenate([edge_affs["between_connected"], edge_dict["edge_affs"]])
            edge_areas["between_connected"] = np.concatenate([edge_areas["between_connected"], edge_dict["edge_areas"]])

    # Initialize an ChunkedGraph instance and write to it
    cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    cg.add_atomic_edges_in_chunks(edge_ids, edge_affs, edge_areas,
                                  isolated_node_ids=isolated_ids,
                                  verbose=verbose)


def _add_layer_thread(args):
    """ Creates abstraction layer """
    table_id, layer_id, chunk_coords, n_threads_per_process = args

    cg = chunkedgraph.ChunkedGraph(table_id=table_id)
    cg.add_layer(layer_id, chunk_coords, n_threads=n_threads_per_process)

