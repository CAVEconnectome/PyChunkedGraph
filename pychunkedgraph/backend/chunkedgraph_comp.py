import numpy as np
import datetime

from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend.utils import column_keys

from multiwrapper import multiprocessing_utils as mu

from typing import Optional, Sequence


def _read_root_rows_thread(args) -> list:
    start_seg_id, end_seg_id, serialized_cg_info, time_stamp = args

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    start_id = cg.get_node_id(segment_id=start_seg_id,
                              chunk_id=cg.root_chunk_id)
    end_id = cg.get_node_id(segment_id=end_seg_id,
                            chunk_id=cg.root_chunk_id)

    rows = cg.read_node_id_rows(
        start_id=start_id,
        end_id=end_id,
        end_id_inclusive=False,
        end_time=time_stamp,
        end_time_inclusive=True)

    root_ids = [k for (k, v) in rows.items()
                if column_keys.Hierarchy.NewParent not in v]

    return root_ids


def get_latest_roots(cg,
                     time_stamp: Optional[datetime.datetime] = None,
                     n_threads: int = 1) -> Sequence[np.uint64]:

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
                                       verbose=False, debug=n_threads == 1)
    else:
        results = mu.multisubprocess_func(_read_root_rows_thread,
                                          multi_args, n_threads=n_threads)

    root_ids = []
    for result in results:
        root_ids.extend(result)

    return np.array(root_ids, dtype=np.uint64)
