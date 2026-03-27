"""ID allocation using cg.id_client."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from google.cloud.bigtable.data.row_filters import CellsRowLimitFilter, RowFilterChain, StripValueTransformerFilter
from kvdbclient.serializers import deserialize_uint64, serialize_uint64_batch

_EXIST_FILTER = RowFilterChain(filters=[
    CellsRowLimitFilter(1),
    StripValueTransformerFilter(True),
])


def batch_create(cg, size_map: dict, root_chunks: set = None) -> dict:
    if root_chunks is None:
        root_chunks = set()

    def _alloc(chunk_id: int) -> tuple:
        count = size_map[chunk_id]
        if chunk_id not in root_chunks:
            return chunk_id, list(cg.id_client.create_node_ids(
                np.uint64(chunk_id), size=count, root_chunk=False,
            ))
        batch_size = count
        new_ids = []
        while len(new_ids) < count:
            candidates = cg.id_client.create_node_ids(
                np.uint64(chunk_id), size=batch_size, root_chunk=True,
            )
            rows = cg.client._read(
                row_keys=serialize_uint64_batch(candidates),
                row_filter=_EXIST_FILTER,
            )
            existing = {deserialize_uint64(k) for k in rows}
            new_ids.extend(set(candidates) - existing)
            batch_size = min(batch_size * 2, 2**16)
        return chunk_id, new_ids[:count]

    result = {}
    n_workers = min(len(size_map), 4)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_alloc, c): c for c in size_map}
        for fut in as_completed(futs):
            cid, ids = fut.result()
            result[cid] = ids
    return result
