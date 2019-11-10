import collections
import numpy as np

from pychunkedgraph.backend.utils import column_keys
from pychunkedgraph.backend import chunkedgraph_exceptions as cg_exceptions

former_parent_col = column_keys.Hierarchy.FormerParent
operation_id_col = column_keys.OperationLogs.OperationID

class SegmentHistory(object):
    def __init__(self, cg, root_id):
        if not cg.is_root(root_id):
            raise cg_exceptions.Forbidden

        self.cg = cg
        self.root_id = root_id

        self._past_log_rows = None

    @property
    def past_log_entries(self):
        if self._past_log_rows is None:
            self._collect_past_edits()

        return self._past_log_rows

    @property
    def past_merge_entries(self):
        log_entries = {}
        for operation_id, log_entry in self.past_log_entries.items():
            if log_entry.is_merge:
                log_entries[operation_id] = log_entry

        return log_entries

    @property
    def past_split_entries(self):
        log_entries = {}
        for operation_id, log_entry in self.past_log_entries.items():
            if not log_entry.is_merge:
                log_entries[operation_id] = log_entry

        return log_entries

    @property
    def last_edit(self):
        row = self.cg.read_node_id_row(node_id=self.root_id)

        if not former_parent_col in row:
            return None
        else:
            former_ids = row[former_parent_col][0].value
            former_row = self.cg.read_node_id_row(node_id=former_ids[0])
            operation_id = former_row[operation_id_col][0].value
            log_entry = LogEntry(*self.cg.read_log_row(operation_id))

        return log_entry

    def _collect_past_edits(self):
        self._past_log_rows = {}

        next_ids = [self.root_id]

        while len(next_ids):
            row_dict = self.cg.read_node_id_rows(node_ids=next_ids)

            next_ids = []

            for row_key, row in row_dict.items():

                # Get former root ids if available
                if former_parent_col in row:
                    former_ids = row[former_parent_col][0].value

                    next_ids.extend(former_ids)

                # Read log row and add it to the dict
                if operation_id_col in row:
                    operation_id = row[operation_id_col][0].value

                    if operation_id in self._past_log_rows:
                        continue
                else:
                    if row_key != self.root_id:
                        raise cg_exceptions.InternalServerError
                    else:
                        continue

                log_row, log_timestamp = self.cg.read_log_row(operation_id)

                self._past_log_rows[operation_id] = LogEntry(log_row,
                                                             log_timestamp)

    def merge_log(self, correct_for_wrong_coord_type=True):
        merge_entries = self.past_merge_entries

        added_edges = []
        added_edge_coords = []
        for _, log_entry in merge_entries.items():
            added_edges.append(log_entry.added_edge)

            coords = log_entry.coordinates

            if correct_for_wrong_coord_type:
                # A little hack because we got the datatype wrong...
                coords = [np.frombuffer(coords[0]),
                          np.frombuffer(coords[1])]
                coords *= self.cg.segmentation_resolution
            added_edge_coords.append(coords)

        return {"merge_edges": added_edges,
                "merge_edge_coords": added_edge_coords}

    def change_log(self):
        user_dict = collections.defaultdict(collections.Counter)

        past_ids = []
        for _, log_entry in self.past_split_entries.items():
            past_ids.extend(log_entry.root_ids)
            user_dict[log_entry.user_id]["n_splits"] += 1

        for _, log_entry in self.past_merge_entries.items():
            past_ids.extend(log_entry.root_ids)
            user_dict[log_entry.user_id]["n_mergers"] += 1

        return {"n_splits": len(self.past_split_entries),
                "n_mergers": len(self.past_merge_entries),
                "user_info": user_dict,
                "operations_ids": np.array(list(self.past_log_entries.keys())),
                "past_ids": past_ids}


class LogEntry(object):
    def __init__(self, row, timestamp):
        self.row = row
        self.timestamp = timestamp

    @property
    def is_merge(self):
        return column_keys.OperationLogs.AddedEdge in self.row

    @property
    def user_id(self):
        return self.row[column_keys.OperationLogs.UserID]

    @property
    def root_ids(self):
        return self.row[column_keys.OperationLogs.RootID]

    @property
    def added_edge(self):
        if not self.is_merge:
            raise cg_exceptions.InternalServerError

        return self.row[column_keys.OperationLogs.AddedEdge]

    @property
    def coordinates(self):
        return np.array([self.row[column_keys.OperationLogs.SourceCoordinate],
                         self.row[column_keys.OperationLogs.SinkCoordinate]])

    def __str__(self):
        log_type = "merge" if self.is_merge else "split"
        return f"{self.user_id},{log_type},{self.root_ids},{self.timestamp}"
        
    def __iter__(self):
        log_type = "merge" if self.is_merge else "split"
        attrs = [self.user_id, log_type, self.root_ids, self.timestamp]
        for attr in attrs:
            yield attr


def get_all_log_entries(cg_instance):
    log_entries = []
    log_rows = cg_instance.read_log_rows()
    for operation_id in range(cg_instance.get_max_operation_id()):
        try:
            log_entries.append(
                LogEntry(
                    log_rows[operation_id],
                    log_rows[operation_id]["timestamp"]
                )
            )
        except KeyError:
            continue
    return log_entries