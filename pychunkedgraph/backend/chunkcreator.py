import glob
import numpy as np
import os
import re
import time
import itertools

from cloudvolume import Storage, storage

# from chunkedgraph import ChunkedGraph
from . import chunkedgraph
from . import multiprocessing_utils as mu
from . import utils


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
    else:
        raise Exception("Could not identify region graph ressource")

    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = utils.dir_from_layer_name(
            utils.layer_name_from_cv_url(cv_st.layer_path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_paths = list(cv_st.list_files())

    file_chunks = np.array_split(file_paths, n_threads * 3)
    multi_args = []
    for i_file_chunk, file_chunk in enumerate(file_chunks):
        multi_args.append([i_file_chunk, cv_url, file_chunk, olduint32])

    # Run multiprocessing
    if n_threads == 1:
        mu.multiprocess_func(_download_and_store_cv_files_thread,
                             multi_args, n_threads=n_threads,
                             verbose=True, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_download_and_store_cv_files_thread,
                                multi_args,
                                n_threads=n_threads)


def _download_and_store_cv_files_thread(args):
    """ Helper thread to download files from google cloud """
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
                print("%d: %d / %d - dt: %.3fs - eta: %.3fs" % (
                chunk_id, i_fp, n_file_paths, dt, eta))

            if "rg2cg" in fp:
                utils.download_and_store_mapping_file(cv_st, fp, olduint32)
            else:
                utils.download_and_store_edge_file(cv_st, fp)


def check_stored_cv_files(dataset_name="basil"):
    """ Tests if all files were downloaded

    :param dataset_name: str
    """
    if "basil" == dataset_name:
        cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
    elif "pinky40" == dataset_name:
        cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
    else:
        raise Exception("Could not identify region graph ressource")

    with storage.SimpleStorage(cv_url) as cv_st:
        dir_path = utils.dir_from_layer_name(
            utils.layer_name_from_cv_url(cv_st.layer_path))

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

        #
        # if "rg2cg" in fp:
        #     utils.download_and_store_mapping_file(cv_st, fp)
        # else:
        #     utils.download_and_store_edge_file(cv_st, fp)
    print("%d files were missing" % c)


def _family_consistency_test_thread(args):
    """ Helper to test family consistency """

    table_id, coord, layer_id = args

    x, y, z = coord

    cg = chunkedgraph.ChunkedGraph(table_id)

    rows = cg.range_read_chunk(x, y, z, layer_id)

    failed_node_ids = []

    time_start = time.time()
    for i_k, k in enumerate(rows.keys()):
        if i_k % 100 == 1:
            dt = time.time() - time_start
            eta = dt / i_k * len(rows) - dt
            print("%d / %d - %.3fs -> %.3fs      " % (i_k, len(rows), dt, eta),
                  end="\r")

        node_id = chunkedgraph.deserialize_node_id(k)
        parent_id = np.frombuffer(rows[k].cells["0"][b'parents'][0].value,
                                  dtype=np.uint64)
        if not node_id in cg.get_children(parent_id):
            failed_node_ids.append([node_id, parent_id])

    return failed_node_ids


def family_consistency_test(table_id, n_threads=64):
    """ Runs a simple test on the WHOLE graph

    tests: id in children(parent(id))

    :param table_id: str
    :param n_threads: int
    :return: dict
        n x 2 per layer
        each failed pair: (node_id, parent_id)
    """

    assert "basil" in table_id

    cg = chunkedgraph.ChunkedGraph(table_id)

    failed_node_id_dict = {}
    for layer_id in range(1, cg.n_layers):
        print("\n\n Layer %d \n\n" % layer_id)

        step = int(cg.fan_out ** np.max([0, layer_id - 2]))
        coords = list(itertools.product(range(0, 8, step),
                                        range(0, 8, step),
                                        range(0, 4, step)))

        multi_args = []
        for coord in coords:
            multi_args.append([table_id, coord, layer_id])

        collected_failed_node_ids = mu.multisubprocess_func(
            _family_consistency_test_thread, multi_args, n_threads=n_threads)

        failed_node_ids = []
        for _failed_node_ids in collected_failed_node_ids:
            failed_node_ids.extend(_failed_node_ids)

        failed_node_id_dict[layer_id] = np.array(failed_node_ids)

        print("\n%d nodes rows failed\n" % len(failed_node_ids))

    return failed_node_id_dict


def create_chunked_graph(table_id=None, cv_url=None, fan_out=2,
                         chunk_size=(512, 512, 64), n_threads=1):
    """ Creates chunked graph from downloaded files

    :param table_id: str
    :param cv_url: str
    :param n_threads: int
    """
    if cv_url is None:
        if "basil" in table_id:
            cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
        elif "pinky40" in table_id:
            cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
        else:
            raise Exception("Could not identify region graph ressource")

    times = []
    time_start = time.time()

    file_paths = np.sort(glob.glob(
        utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*"))

    file_path_blocks = np.array_split(file_paths, n_threads * 3)

    multi_args = []
    for fp_block in file_path_blocks:
        multi_args.append([fp_block, table_id, chunk_size])

    if n_threads == 1:
        results = mu.multiprocess_func(
            _preprocess_chunkedgraph_data_thread, multi_args,
            n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(
            _preprocess_chunkedgraph_data_thread, multi_args,
            n_threads=n_threads)

    mapping_paths = np.array([])
    mapping_chunk_ids = np.array([]).reshape(-1, 3)
    in_chunk_paths = np.array([])
    in_chunk_ids = np.array([]).reshape(-1, 3)
    between_chunk_paths = np.array([])
    between_chunk_ids = np.array([]).reshape(-1, 2, 3)

    for result in results:
        mapping_paths = np.concatenate([mapping_paths, result[0]])
        mapping_chunk_ids = np.concatenate([mapping_chunk_ids, result[1]])
        in_chunk_paths = np.concatenate([in_chunk_paths, result[2]])
        in_chunk_ids = np.concatenate([in_chunk_ids, result[3]])
        between_chunk_paths = np.concatenate([between_chunk_paths, result[4]])
        between_chunk_ids = np.concatenate([between_chunk_ids, result[5]])

    times.append(["Preprocessing", time.time() - time_start])
    time_start = time.time()

    multi_args = []

    in_chunk_id_blocks = np.array_split(in_chunk_ids,
                                        max(1, mu.cpu_count()))
    cumsum = 0
    for in_chunk_id_block in in_chunk_id_blocks:
        multi_args.append([between_chunk_ids, between_chunk_paths,
                           in_chunk_id_block, mapping_chunk_ids, mapping_paths,
                           cumsum])
        cumsum += len(in_chunk_id_block)

    # Run multiprocessing
    if n_threads == 1:
        results = mu.multiprocess_func(
            _between_chunk_masks_thread, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        results = mu.multiprocess_func(
            _between_chunk_masks_thread, multi_args, n_threads=n_threads)

    n_layers = int(
        np.ceil(chunkedgraph.log_n(np.max(in_chunk_ids) + 1, fan_out))) + 2

    print("N layers: %d" % n_layers)

    cg = chunkedgraph.ChunkedGraph(table_id=table_id, n_layers=n_layers,
                                   fan_out=fan_out, chunk_size=chunk_size,
                                   is_new=True)

    # Fill lowest layer and create first abstraction layer
    # Create arguments for multiprocessing

    multi_args = []
    for result in results:
        offset, between_chunk_paths_out_masked, \
        between_chunk_paths_in_masked, masked_mapping_paths = result

        for i_chunk in range(len(between_chunk_paths_out_masked)):
            multi_args.append([table_id,
                               in_chunk_paths[offset + i_chunk],
                               between_chunk_paths_in_masked[i_chunk],
                               between_chunk_paths_out_masked[i_chunk],
                               masked_mapping_paths[i_chunk]])

    times.append(["Data sorting", time.time() - time_start])
    time_start = time.time()

    # Run multiprocessing
    if n_threads == 1:
        mu.multiprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads,
            verbose=True, debug=n_threads == 1)
    else:
        mu.multisubprocess_func(
            _create_atomic_layer_thread, multi_args, n_threads=n_threads)

    times.append(["Layers 1 + 2", time.time() - time_start])

    # Fill higher abstraction layers
    child_chunk_ids = in_chunk_ids.copy()
    for layer_id in range(3, n_layers + 1):
        time_start = time.time()

        print("\n\n\n --- LAYER %d --- \n\n\n" % layer_id)

        parent_chunk_ids = child_chunk_ids // cg.fan_out ** (layer_id - 2)

        u_pcids, inds = np.unique(parent_chunk_ids,
                                  axis=0, return_inverse=True)

        if len(u_pcids) > n_threads:
            n_threads_per_process = 1
        else:
            n_threads_per_process = int(np.ceil(n_threads / len(u_pcids)))

        multi_args = []
        for ind in range(len(u_pcids)):
            multi_args.append([table_id, layer_id,
                               child_chunk_ids[inds == ind].astype(np.int),
                               n_threads_per_process])

        # print(multi_args)
        # print(u_pcids, len(multi_args))
        # print(layer_id, np.unique(child_chunk_ids), np.unique(parent_chunk_ids))
        child_chunk_ids = u_pcids * cg.fan_out ** (layer_id - 2)
        # print(layer_id, np.unique(child_chunk_ids), np.unique(parent_chunk_ids))

        # Run multiprocessing
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
    """ Reads downloaded files and formats them """
    file_paths, table_id, chunk_size = args

    chunk_size = np.array(list(chunk_size))

    mapping_paths = np.array([])
    mapping_chunk_ids = np.array([], dtype=np.int).reshape(-1, 3)
    in_chunk_paths = np.array([])
    in_chunk_ids = np.array([], dtype=np.int).reshape(-1, 3)
    between_chunk_paths = np.array([])
    between_chunk_ids = np.array([], dtype=np.int).reshape(-1, 2, 3)

    # Read file paths - gather chunk ids and in / out properties
    for i_fp, fp in enumerate(file_paths):
        file_name = os.path.basename(fp).split(".")[0]

        # Read coordinates from file path
        x1, x2, y1, y2, z1, z2 = np.array(re.findall("[\d]+", file_name),
                                          dtype=np.int)[:6]
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
                chunk_coord[s_c] += 1 - chunk_size[s_c]
                chunk1_id = np.array(chunk_coord / chunk_size, dtype=np.int8)
                chunk_coord[s_c] += chunk_size[s_c]
                chunk2_id = np.array(chunk_coord / chunk_size, dtype=np.int8)

                between_chunk_ids = np.concatenate([between_chunk_ids,
                                                    np.array(
                                                        [chunk1_id, chunk2_id])[
                                                        None]])
                between_chunk_paths = np.concatenate(
                    [between_chunk_paths, [fp]])
            else:
                continue
        else:
            chunk_coord = np.array(c / chunk_size, dtype=np.int8)

            if "rg2cg" in file_name:
                mapping_paths = np.concatenate([mapping_paths, [fp]])
                mapping_chunk_ids = np.concatenate(
                    [mapping_chunk_ids, chunk_coord[None]])
            elif "atomicedges" in file_name:
                in_chunk_ids = np.concatenate([in_chunk_ids, chunk_coord[None]])
                in_chunk_paths = np.concatenate([in_chunk_paths, [fp]])

    return mapping_paths, mapping_chunk_ids, in_chunk_paths, in_chunk_ids, \
           between_chunk_paths, between_chunk_ids


def _between_chunk_masks_thread(args):
    between_chunk_ids, between_chunk_paths, in_chunk_id_block, \
    mapping_chunk_ids, mapping_paths, offset = args

    between_chunk_paths_out_masked = []
    between_chunk_paths_in_masked = []
    masked_mapping_paths = []

    n_blocks = len(in_chunk_id_block)

    time_start = time.time()
    for i_in_chunk_id, in_chunk_id in enumerate(in_chunk_id_block):
        # if i_in_chunk_id % 500 == 1:
        #     dt = time.time() - time_start
        #     eta = dt / i_in_chunk_id * n_blocks - dt
        #     print("%d: %d / %d - dt: %.3fs - eta: %.3fs" %
        #           (i_in_chunk_id + offset, i_in_chunk_id, n_blocks, dt, eta))

        out_paths_mask = np.sum(np.abs(between_chunk_ids[:, 0] -
                                       in_chunk_id), axis=1) == 0
        in_paths_masks = np.sum(np.abs(between_chunk_ids[:, 1] -
                                       in_chunk_id), axis=1) == 0

        mapping_path_masks = np.sum(np.abs(mapping_chunk_ids -
                                           in_chunk_id), axis=1) == 0

        between_chunk_paths_out_masked.append(
            between_chunk_paths[out_paths_mask])
        between_chunk_paths_in_masked.append(
            between_chunk_paths[in_paths_masks])
        masked_mapping_paths.append(mapping_paths[mapping_path_masks][0])

    return offset, between_chunk_paths_out_masked, \
           between_chunk_paths_in_masked, masked_mapping_paths


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
        cross_edge_ids = np.concatenate(
            [cross_edge_ids, this_edge_ids[:, [1, 0]]])
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
    isolated_node_ids = mappings[:, 1][~np.in1d(mappings[:, 1], np.concatenate(
        [np.unique(edge_ids), cross_edge_ids[:, 0]]))]

    # Initialize an ChunkedGraph instance and write to it
    cg = chunkedgraph.ChunkedGraph(table_id=table_id)
    cg.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                  edge_affs, cross_edge_affs,
                                  isolated_node_ids, cg2rg, rg2cg)


def _add_layer_thread(args):
    """ Creates abstraction layer """
    table_id, layer_id, chunk_coords, n_threads_per_process = args

    cg = chunkedgraph.ChunkedGraph(table_id=table_id)
    cg.add_layer(layer_id, chunk_coords, n_threads=n_threads_per_process)

