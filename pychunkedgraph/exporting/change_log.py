import dill
import numpy as np
import time
from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.backend.utils import column_keys
from pychunkedgraph.backend import chunkedgraph


def export_changelog(cg, path=None):
    """ Exports all changes to binary dill file

    :param cg: ChunkedGraph instance
    :param path: str
    :return: bool
    """

    operations = cg.read_node_id_rows(start_id=np.uint64(0),
                                      end_id=cg.get_max_operation_id(),
                                      end_id_inclusive=True)

    if path is not None:
        with open(path, "wb") as f:
            dill.dump(operations, f)
    else:
        return operations


def load_changelog(path):
    """ Loads stored changelog

    :param path: str
    :return:
    """

    with open(path, "rb") as f:
        operations = dill.load(f)

    # Dill can marshall the `serializer` functions used for each column, but .
    # their address won't match anymore, which breaks the hash lookup for
    # `_Column`s. Hence, we simply create new `_Column`s from the old
    # `family_id` and `key`
    for operation_id, column_dict in operations.items():
        operations[operation_id] = \
            {column_keys.from_key(k.family_id, k.key): v
             for (k, v) in column_dict.items()}

    return operations


def _choose_decent_center_coord(coords):
    com = np.median(coords, axis=0).astype(np.int)

    if np.any(np.linalg.norm(com - coords, axis=1) == 0):
        return com
    else:
        return coords[int(len(coords) / 2)]


def lookup_sv_coords(cg, sv_ids):
    sv_coords = np.zeros([len(sv_ids), 3], dtype=np.int)
    chunk_ids = []
    for sv_id in sv_ids:
        chunk_ids.append(cg.get_chunk_id(sv_id))

    u_chunk_ids, sv_idx = np.unique(chunk_ids, return_inverse=True)

    for i_chunk_id, chunk_id in enumerate(u_chunk_ids):
        sv_m = sv_idx == i_chunk_id
        chunk_sv_ids = sv_ids[sv_m]

        bb_start = (cg.get_chunk_coordinates(chunk_id) *
                    cg.chunk_size).astype(np.int)
        bb_end = (bb_start + cg.chunk_size).astype(np.int)

        ws_seg = cg.cv[bb_start[0]: bb_end[0],
                       bb_start[1]: bb_end[1],
                       bb_start[2]: bb_end[2]].squeeze()

        this_sv_coords = []
        for sv_id in chunk_sv_ids:
            ws_seg_coords = np.array(np.where(ws_seg == sv_id)).T + bb_start
            this_sv_coords.append(_choose_decent_center_coord(ws_seg_coords))

        sv_coords[sv_m] = np.array(this_sv_coords)

    return sv_coords


def process_single_changelog_entry(cg, cl_entry, recalc_coords=True):
    # Determine whether an edit is split or merge
    is_split = column_keys.OperationLogs.RemovedEdge in cl_entry

    # Get supervoxel ids
    sink_ids = cl_entry[column_keys.OperationLogs.SinkID][0].value
    source_ids = cl_entry[column_keys.OperationLogs.SourceID][0].value

    # Extract coordinates from supervoxels
    if recalc_coords:
        sv_coords = lookup_sv_coords(cg, np.concatenate([sink_ids, source_ids]))
        sink_coords = sv_coords[:len(sink_ids)]
        source_coords = sv_coords[len(sink_ids):]
    else:
        sink_coords = np.frombuffer(cl_entry[column_keys.OperationLogs.SinkCoordinate][0].value).reshape(-1, 3) * cg.segmentation_resolution
        sink_coords = sink_coords.astype(np.int)
        source_coords = np.frombuffer(cl_entry[column_keys.OperationLogs.SourceCoordinate][0].value).reshape(-1, 3) * cg.segmentation_resolution
        source_coords = source_coords.astype(np.int)

    # User id
    user_id = cl_entry[column_keys.OperationLogs.UserID][0].value

    # Time stamp
    time_stamp = cl_entry[column_keys.OperationLogs.UserID][0].timestamp

    cl_entry_p = {"is_split": is_split,
                  "sink_coords": sink_coords,
                  "source_coords": source_coords,
                  "sink_sv_ids": sink_ids,
                  "source_sv_ids": source_ids,
                  "timestamp": time_stamp,
                  "user_id": "default"}
    return cl_entry_p


def _process_changelog_entries(args):
    cg_info, cl_keys, cl_path, recalc_coords = args

    cl = load_changelog(cl_path)
    cl_p = {}
    cg = chunkedgraph.ChunkedGraph(**cg_info)

    times = []
    for cl_key in cl_keys:
        time_start = time.time()
        cl_p[cl_key] = process_single_changelog_entry(cg, cl[cl_key],
                                                      recalc_coords=recalc_coords)
        times.append(time.time() - time_start)

    print(np.mean(times), np.std(times))

    return cl_p


def reformat_changelog(cg, cl_path, out_path=None, n_threads=10, n_limit=None,
                       recalc_coords=True):
    cl = load_changelog(cl_path)

    cg_info = cg.get_serialized_info()
    del cg_info["credentials"]

    cl_keys = list(cl.keys())
    if n_limit is not None:
        cl_keys = cl_keys[:n_limit]

    cl_key_blocks = np.array_split(cl_keys, n_threads * 3)

    multi_args = []
    for cl_key_block in cl_key_blocks:
        multi_args.append([cg_info, cl_key_block, cl_path, recalc_coords])

    results = mu.multisubprocess_func(_process_changelog_entries, multi_args,
                                      n_threads=n_threads)

    cl_p = {}
    for result in results:
        cl_p.update(result)

    if out_path is not None:
        with open(out_path, "wb") as f:
            dill.dump(cl_p, f)
    else:
        return cl_p


def get_log_diff(log_old, log_new):
    """ Computes a simple difference between two logs

    :param log_old: dict
    :param log_new: dict
    :return: dict
    """
    log = log_new.copy()

    for k in log_old:
        del log[k]

    return log
