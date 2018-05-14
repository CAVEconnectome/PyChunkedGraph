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


def download_and_store_cv_files(cv_url="gs://nkem/basil_4k_oldnet/region_graph/"):
    with storage.SimpleStorage(cv_url) as cv_st:
        file_paths = cv_st.list_files()

        for fp in file_paths:
            print(fp)
            if "rg2cg" in fp:
                utils.download_and_store_mapping_file(cv_st, fp)
            else:
                utils.download_and_store_edge_file(cv_st, fp)


def create_chunked_graph(cv_url="gs://nkem/basil_4k_oldnet/region_graph/",
                         dev_mode=False, table_id=None, nb_cpus=1):

    # Currently no multiprocessing...
    # assert nb_cpus == 1

    file_paths = np.sort(glob.glob(utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*"))

    cg = chunkedgraph.ChunkedGraph(dev_mode=dev_mode, table_id=table_id)

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

        multi_args.append([dev_mode,
                           table_id,
                           chunk_path,
                           between_chunk_paths[in_paths_mask],
                           between_chunk_paths[out_paths_mask],
                           mapping_paths[i_chunk]])

    # Run multiprocessing
    # storage.S3_POOL.reset_pool()
    # storage.GC_POOL["neuroglancer"].reset_pool()
    if nb_cpus == 1:
        multiprocessing_utils.multiprocess_func(_create_atomic_layer_thread,
                                                multi_args, nb_cpus=nb_cpus,
                                                verbose=True, debug=nb_cpus==1)
    else:
        multiprocessing_utils.multisubprocess_func(_create_atomic_layer_thread,
                                                   multi_args,
                                                   n_subprocesses=nb_cpus)

    # Fill higher abstraction layers
    layer_id = 2
    child_chunk_ids = in_chunk_ids.copy()
    last_run = False
    while not last_run:
        layer_id += 1

        print("\n\n\n --- LAYER %d --- \n\n\n" % layer_id)

        if len(child_chunk_ids) == 1:
            last_run = True

        parent_chunk_ids = child_chunk_ids // cg.fan_out ** (layer_id - 2)

        u_pcids, inds = np.unique(parent_chunk_ids,
                                  axis=0, return_inverse=True)

        multi_args = []
        for ind in range(len(u_pcids)):
            multi_args.append([dev_mode, table_id, layer_id, child_chunk_ids[inds == ind]])

        child_chunk_ids = u_pcids * cg.fan_out ** (layer_id - 2)

        # Run multiprocessing
        if nb_cpus == 1:
            multiprocessing_utils.multiprocess_func(_add_layer_thread, multi_args,
                                                    nb_cpus=nb_cpus, verbose=True,
                                                    debug=nb_cpus==1)
        else:
            multiprocessing_utils.multisubprocess_func(_add_layer_thread,
                                                       multi_args,
                                                       n_subprocesses=nb_cpus)


def _create_atomic_layer_thread(args):
    """ Fills lowest layer and create first abstraction layer """
    # Reset connection pool to make cloud-volume compatible with multiprocessing
    # storage.reset_connection_pools()

    # Load args
    dev_mode, table_id, chunk_path, in_paths, out_paths, mapping_path = args
    # edge_ids, edge_affs, cross_edge_ids, cross_edge_affs = args

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

    # Initialize an ChunkedGraph instance and write to it
    cg = chunkedgraph.ChunkedGraph(dev_mode=dev_mode, table_id=table_id)
    cg.add_atomic_edges_in_chunks(edge_ids, cross_edge_ids,
                                  edge_affs, cross_edge_affs,
                                  cg2rg, rg2cg)


def _add_layer_thread(args):
    dev_mode, table_id, layer_id, chunk_coords = args

    cg = chunkedgraph.ChunkedGraph(dev_mode=dev_mode, table_id=table_id)
    cg.add_layer(layer_id, chunk_coords)

