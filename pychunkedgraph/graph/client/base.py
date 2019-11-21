from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Iterable

import numpy as np

from ..meta import ChunkedGraphMeta


# TODO design api
# 1. create / overwrite
# 2. a counter to generate unique ids (IDs api?)
# 3. store metadata
# 4. read/write rows api


class Client(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def create_graph(self, graph_meta: ChunkedGraphMeta) -> None:
        """Create graph and store associated meta"""

    @abstractmethod
    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """
        Read nodes and their properties.
        A range of node IDs or specific node IDs.
        """

    @abstractmethod
    def read_node(
        self,
        node_id: np.uint64,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single node and it's properties"""



    def mutate_row(
        self,
        row_key: bytes,
        val_dict: Dict[column_keys._Column, Any],
        time_stamp: Optional[datetime.datetime] = None,
        isbytes: bool = False,
    ) -> bigtable.row.Row:
        """ Mutates a single row
        :param row_key: serialized bigtable row key
        :param val_dict: Dict[column_keys._TypedColumn: bytes]
        :param time_stamp: None or datetime
        :return: list
        """
        row = self.table.row(row_key)
        for column, value in val_dict.items():
            if not isbytes:
                value = column.serialize(value)
            row.set_cell(
                column_family_id=column.family_id,
                column=column.key,
                value=value,
                timestamp=time_stamp,
            )
        return row

    def bulk_write(
        self,
        rows: Iterable[bigtable.row.DirectRow],
        root_ids: Optional[Union[np.uint64, Iterable[np.uint64]]] = None,
        operation_id: Optional[np.uint64] = None,
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """ Writes a list of mutated rows in bulk
        WARNING: If <rows> contains the same row (same row_key) and column
        key two times only the last one is effectively written to the BigTable
        (even when the mutations were applied to different columns)
        --> no versioning!
        :param rows: list
            list of mutated rows
        :param root_ids: list if uint64
        :param operation_id: uint64 or None
            operation_id (or other unique id) that *was* used to lock the root
            the bulk write is only executed if the root is still locked with
            the same id.
        :param slow_retry: bool
        :param block_size: int
        """
        if slow_retry:
            initial = 5
        else:
            initial = 1

        retry_policy = Retry(
            predicate=if_exception_type(
                (Aborted, DeadlineExceeded, ServiceUnavailable)
            ),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=LOCK_EXPIRED_TIME_DELTA.seconds,
        )

        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.check_and_renew_root_locks(root_ids, operation_id):
                raise cg_exceptions.LockError(
                    f"Root lock renewal failed for operation ID {operation_id}"
                )

        for i_row in range(0, len(rows), block_size):
            status = self.table.mutate_rows(
                rows[i_row : i_row + block_size], retry=retry_policy
            )
            if not all(status):
                raise cg_exceptions.ChunkedGraphError(
                    f"Bulk write failed for operation ID {operation_id}"
                )

    def _execute_read_thread(self, row_set_and_filter: Tuple[RowSet, RowFilter]):
        row_set, row_filter = row_set_and_filter
        if not row_set.row_keys and not row_set.row_ranges:
            # Check for everything falsy, because Bigtable considers even empty
            # lists of row_keys as no upper/lower bound!
            return {}

        range_read = self.table.read_rows(row_set=row_set, filter_=row_filter)
        res = {v.row_key: partial_row_data_to_column_dict(v) for v in range_read}
        return res

    def _execute_read(
        self, row_set: RowSet, row_filter: RowFilter = None
    ) -> Dict[bytes, Dict[column_keys._Column, bigtable.row_data.PartialRowData]]:
        """ Core function to read rows from Bigtable. Uses standard Bigtable retry logic
        :param row_set: BigTable RowSet
        :param row_filter: BigTable RowFilter
        :return: Dict[bytes, Dict[column_keys._Column, bigtable.row_data.PartialRowData]]
        """

        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now
        max_row_key_count = 20000
        n_subrequests = max(1, int(np.ceil(len(row_set.row_keys) / max_row_key_count)))
        n_threads = min(n_subrequests, 2 * mu.n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[
                i * max_row_key_count : (i + 1) * max_row_key_count
            ]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges

        responses = mu.multithread_func(
            self._execute_read_thread,
            params=((r, row_filter) for r in row_sets),
            debug=n_threads == 1,
            n_threads=n_threads,
        )

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)

        return combined_response

