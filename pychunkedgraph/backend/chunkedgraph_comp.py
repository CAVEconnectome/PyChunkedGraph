import numpy as np
import datetime

from pychunkedgraph.backend import key_utils, chunkedgraph

from multiwrapper import multiprocessing_utils as mu
from pychunkedgraph.backend.chunkedgraph_utils import compute_indices_pandas, \
    compute_bitmasks, get_google_compatible_time_stamp, \
    get_inclusive_time_range_filter, get_max_time, \
    combine_cross_chunk_edge_dicts, time_min

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _read_root_rows_thread(args) -> list:
    start_seg_id, end_seg_id, serialized_cg_info, time_stamp = args

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    time_filter = get_inclusive_time_range_filter(end=time_stamp)

    start_id = cg.get_node_id(segment_id=start_seg_id,
                              chunk_id=cg.root_chunk_id)
    end_id = cg.get_node_id(segment_id=end_seg_id, chunk_id=cg.root_chunk_id)
    range_read = cg.table.read_rows(
        start_key=key_utils.serialize_uint64(start_id),
        end_key=key_utils.serialize_uint64(end_id),
        # allow_row_interleaving=True,
        end_inclusive=False,
        filter_=time_filter)

    range_read.consume_all()
    rows = range_read.rows

    root_ids = []
    for row_id, row_data in rows.items():
        row_keys = row_data.cells[cg.family_id]

        if not key_utils.serialize_key("new_parents") in row_keys:
            root_ids.append(key_utils.deserialize_uint64(row_id))

    return root_ids


def get_latest_roots(cg,
                     time_stamp: Optional[datetime.datetime] = get_max_time(),
                     n_threads: int = 1) -> Sequence[np.uint64]:

    # Comply to resolution of BigTables TimeRange
    time_stamp = get_google_compatible_time_stamp(time_stamp, round_up=False)

    # Create filters: time and id range
    max_seg_id = cg.get_max_seg_id(cg.root_chunk_id) + 1

    n_blocks = np.min([n_threads * 3 + 1, max_seg_id])
    seg_id_blocks = np.linspace(1, max_seg_id, n_blocks, dtype=np.uint64)

    cg_serialized_info = cg.get_serialized_info()

    if n_threads > 1:
        del cg_serialized_info["credentials"]

    multi_args = []
    for i_id_block in range(0, len(seg_id_blocks) - 1):
        multi_args.append([seg_id_blocks[i_id_block],
                           seg_id_blocks[i_id_block + 1],
                           cg_serialized_info, time_stamp])

    # Run parallelizing
    if n_threads == 1:
        results = mu.multiprocess_func(_read_root_rows_thread,
                                       multi_args, n_threads=n_threads,
                                       verbose=False, debug=n_threads==1)
    else:
        results = mu.multisubprocess_func(_read_root_rows_thread,
                                          multi_args, n_threads=n_threads)

    root_ids = []
    for result in results:
        root_ids.extend(result)

    return np.array(root_ids, dtype=np.uint64)