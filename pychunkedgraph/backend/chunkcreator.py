import glob
import numpy as np
import os
import re
import time

from cloudvolume import Storage, storage

# from chunkedgraph import ChunkedGraph
from . import chunkedgraph
from . import multiprocessing_utils
from . import utils


# def download_and_store_cv_files(cv_url="gs://nkem/basil_4k_oldnet/region_graph/"):
def download_and_store_cv_files(cv_url="gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/",
                                n_threads=10, olduint32=False):
    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_st.layer_path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_paths = list(cv_st.list_files())

    file_chunks = np.array_split(file_paths, n_threads * 3)
    multi_args = []
    for i_file_chunk, file_chunk in enumerate(file_chunks):
        multi_args.append([i_file_chunk, cv_url, file_chunk, olduint32])

    # Run multiprocessing
    if n_threads == 1:
        multiprocessing_utils.multiprocess_func(_download_and_store_cv_files_thread,
                                                multi_args, n_threads=n_threads,
                                                verbose=True, debug=n_threads==1)
    else:
        multiprocessing_utils.multisubprocess_func(_download_and_store_cv_files_thread,
                                                   multi_args,
                                                   n_threads=n_threads)


def _download_and_store_cv_files_thread(args):
    chunk_id, cv_url, file_paths, olduint32 = args

    # Reset connection pool to make cloud-volume compatible with multiprocessing
    storage.reset_connection_pools()

    n_file_paths = len(file_paths)
    time_start = time.time()
    with storage.SimpleStorage(cv_url) as cv_st:
        for i_fp, fp in enumerate(file_paths):
            if i_fp % 100 == 1:
                dt = time.time() - time_start
                eta = dt / i_fp * n_file_paths - dt
                print("%d: %d / %d - dt: %.3fs - eta: %.3fs" % (chunk_id, i_fp, n_file_paths, dt, eta))

            if "rg2cg" in fp:
                utils.download_and_store_mapping_file(cv_st, fp, olduint32)
            else:
                utils.download_and_store_edge_file(cv_st, fp)


def check_stored_cv_files(cv_url="gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"):
    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_st.layer_path))

        file_paths = list(cv_st.list_files())

    c = 0
    n_file_paths = len(file_paths)
    time_start = time.time()
    for i_fp, fp in enumerate(file_paths):
        if i_fp % 1000 == 1:
            dt = time.time() - time_start
            eta = dt / i_fp * n_file_paths - dt
            print("%d / %d - dt: %.3fs - eta: %.3fs" % (i_fp, n_file_paths, dt, eta))

        if not os.path.exists(dir_path + fp[:-4] + ".h5"):
            # print(dir_path + fp[:-4] + ".h5")
            c += 1

        #
        # if "rg2cg" in fp:
        #     utils.download_and_store_mapping_file(cv_st, fp)
        # else:
        #     utils.download_and_store_edge_file(cv_st, fp)
    print(c)


def create_chunked_graph(table_id=None, cv_url=None, n_threads=1):
    if cv_url is None:
        if "basil" in table_id:
            cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
        elif "pinky40" in table_id:
            cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
        else:
            raise Exception("Could not identify region graph ressource")

    times = []
    time_start = time.time()

    file_paths = np.sort(glob.glob(utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*"))

    cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    multi_args = []

    mapping_paths = []
    in_chunk_paths = []
    in_chunk_ids = []
    between_chunk_paths = []
    between_chunk_ids = []

    # Read file paths - gather chunk ids and in / out properties
    for i_fp, fp in enumerate(file_paths):
        file_name = os.path.basename(fp).split(".")[0]

        # Read coordinates from file path
        x1, x2, y1, y2, z1, z2 = np.array(re.findall("[\d]+", file_name), dtype=np.int)[:6]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        d = np.array([dx, dy, dz])
        c = np.array([x1, y1, z1])

        # if there is a 2 in d then the file contains edges that cross chunks
        if 2 in d:
            if "atomicedges" in file_name:
                s_c = np.where(d == 2)[0]
                chunk_coord = c.copy()
                chunk_coord[s_c] += 1 - cg.chunk_size[s_c]
                chunk1_id = np.array(chunk_coord / cg.chunk_size, dtype=np.int8)
                chunk_coord[s_c] += cg.chunk_size[s_c]
                chunk2_id = np.array(chunk_coord / cg.chunk_size, dtype=np.int8)

                between_chunk_ids.append([chunk1_id, chunk2_id])
                between_chunk_paths.append(fp)
            else:
                continue
        else:
            if "rg2cg" in file_name:
                mapping_paths.append(fp)
            elif "atomicedges" in file_name:
                chunk_coord = c.copy()
                in_chunk_ids.append(np.array(chunk_coord / cg.chunk_size, dtype=np.int8))
                in_chunk_paths.append(fp)

    in_chunk_ids = np.array(in_chunk_ids)
    in_chunk_paths = np.array(in_chunk_paths)
    mapping_paths = np.array(mapping_paths)
    between_chunk_ids = np.array(between_chunk_ids)
    between_chunk_paths = np.array(between_chunk_paths)

    # Fill lowest layer and create first abstraction layer
    # Create arguments for multiprocessing
    for i_chunk, chunk_path in enumerate(in_chunk_paths):
        out_paths_mask = np.sum(np.abs(between_chunk_ids[:, 0] - in_chunk_ids[i_chunk]), axis=1) == 0
        in_paths_mask = np.sum(np.abs(between_chunk_ids[:, 1] - in_chunk_ids[i_chunk]), axis=1) == 0

        multi_args.append([table_id,
                           chunk_path,
                           between_chunk_paths[in_paths_mask],
                           between_chunk_paths[out_paths_mask],
                           mapping_paths[i_chunk]])

    times.append(["Preprocessing", time.time() - time_start])
    time_start = time.time()

    # Run multiprocessing
    if n_threads == 1:
        multiprocessing_utils.multiprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads==1)
    else:
        multiprocessing_utils.multisubprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads)

    times.append(["Layers 1 + 2", time.time() - time_start])

    # Fill higher abstraction layers
    layer_id = 2
    child_chunk_ids = in_chunk_ids.copy()
    last_run = False
    while not last_run:
        time_start = time.time()

        layer_id += 1

        print("\n\n\n --- LAYER %d --- \n\n\n" % layer_id)

        parent_chunk_ids = child_chunk_ids // cg.fan_out ** (layer_id - 2)

        # print(parent_chunk_ids)
        # print(child_chunk_ids)

        u_pcids, inds = np.unique(parent_chunk_ids,
                                  axis=0, return_inverse=True)

        multi_args = []
        for ind in range(len(u_pcids)):
            multi_args.append([table_id, layer_id, child_chunk_ids[inds == ind]])

        if len(child_chunk_ids) == 1:
            last_run = True

        child_chunk_ids = u_pcids * cg.fan_out ** (layer_id - 2)

        # Run multiprocessing
        if n_threads == 1:
            multiprocessing_utils.multiprocess_func(
                _add_layer_thread, multi_args, n_threads=n_threads, verbose=True,
                debug=n_threads==1)
        else:
            multiprocessing_utils.multisubprocess_func(
                _add_layer_thread, multi_args, n_threads=n_threads)

        times.append(["Layer %d" % layer_id, time.time() - time_start])

    for time_entry in times:
        print("%s: %.2fs = %.2fmin = %.2fh" % (time_entry[0], time_entry[1],
                                               time_entry[1] / 60,
                                               time_entry[1] / 3600))


def _create_atomic_layer_thread(args):
    """ Fills lowest layer and create first abstraction layer """
    # Load args
    table_id, chunk_path, in_paths, out_paths, mapping_path = args

    # Load edge information
    edge_ids, edge_affs = utils.read_edge_file_h5(chunk_path)
    cross_edge_ids = np.array([], dtype=np.uint64).reshape(0, 2)
    cross_edge_affs = np.array([], dtype=np.float32)

    for fp in in_paths:
        this_edge_ids, this_edge_affs = utils.read_edge_file_h5(fp)

        # Cross edges are always ordered to point OUT of the chunk
        cross_edge_ids = np.concatenate([cross_edge_ids, this_edge_ids[:, [1, 0]]])
        cross_edge_affs = np.concatenate([cross_edge_affs, this_edge_affs])

    for fp in out_paths:
        this_edge_ids, this_edge_affs = utils.read_edge_file_h5(fp)

        cross_edge_ids = np.concatenate([cross_edge_ids, this_edge_ids])
        cross_edge_affs = np.concatenate([cross_edge_affs, this_edge_affs])

    # Load mapping between region and chunkedgraph
    mappings = utils.read_mapping_h5(mapping_path)
    cg2rg = dict(zip(mappings[:, 1], mappings[:, 0]))
    rg2cg = dict(zip(mappings[:, 0], mappings[:, 1]))

    # Get isolated nodes
    isolated_node_ids = mappings[:, 1][~np.in1d(mappings[:, 1], np.concatenate(edge_ids[:, 0], cross_edge_ids[:, 0]))]

    # Initialize an ChunkedGraph instance and write to it
    cg = chunkedgraph.ChunkedGraph(table_id=table_id)
    cg.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                  edge_affs, cross_edge_affs,
                                  isolated_node_ids, cg2rg, rg2cg)


def _add_layer_thread(args):
    table_id, layer_id, chunk_coords = args

    cg = chunkedgraph.ChunkedGraph(table_id=table_id)
    cg.add_layer(layer_id, chunk_coords)

