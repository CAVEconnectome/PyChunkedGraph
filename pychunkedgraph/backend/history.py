import collections
import numpy as np
import datetime
import pandas as pd

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
        self._root_id_lookup_vec = None
        self._original_root_id_lookup_vec = None
        self._edited_sv_ids = None
        self._edit_timestamps = None
        self._tabular_changelog = None

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

    @property
    def edit_timestamps(self):
        if self._edit_timestamps is None:
            self._collect_edit_timestamps()
        return self._edit_timestamps

    @property
    def edited_sv_ids(self):
        if self._edited_sv_ids is None:
            self._collect_edited_sv_ids()

        return self._edited_sv_ids

    @property
    def root_id_lookup_vec(self):
        if self._root_id_lookup_vec is None:
            root_id_lookup_dict = dict(zip(self.edited_sv_ids,
                                           self.cg.get_roots(self.edited_sv_ids,
                                                             time_stamp=np.max(self.edit_timestamps) +
                                                                        datetime.timedelta(seconds=.01))))
            self._root_id_lookup_vec = np.vectorize(root_id_lookup_dict.get)

        return self._root_id_lookup_vec

    @property
    def original_root_id_lookup_vec(self):
        if self._original_root_id_lookup_vec is None:
            root_id_lookup_dict = dict(zip(self.edited_sv_ids,
                                           self.cg.get_roots(self.edited_sv_ids,
                                                             time_stamp=np.min(self.edit_timestamps) -
                                                                        datetime.timedelta(seconds=.01))))
            self._original_root_id_lookup_vec = np.vectorize(root_id_lookup_dict.get)

        return self._original_root_id_lookup_vec

    @property
    def tabular_changelog(self):
        if self._tabular_changelog is None:
            self._build_tabular_changelog()

        return self._tabular_changelog

    def _collect_edit_timestamps(self):
        self._edit_timestamps = []
        for entry_id, entry in self.past_log_entries.items():
            self._edit_timestamps.append(entry.timestamp)

        self._edit_timestamps = np.array(self._edit_timestamps)

    def _collect_edited_sv_ids(self):
        self._edited_sv_ids = []
        for entry_id, entry in self.past_log_entries.items():
            self._edited_sv_ids.extend(entry.edges_failsafe)

        self._edited_sv_ids = np.array(self._edited_sv_ids)


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
                if operation_id_col in row and row_key != self.root_id:
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

    def _build_tabular_changelog(self):
        is_merge_list = []
        is_in_neuron_list = []
        is_relevant_list = []
        timestamp_list = []
        user_list = []
        before_root_ids_list = []
        after_root_ids_list = []

        entry_ids = list(self.past_log_entries.keys())
        for entry_id in np.sort(entry_ids):
            entry = self.past_log_entries[entry_id]

            is_merge_list.append(entry.is_merge)
            timestamp_list.append(entry.timestamp)
            user_list.append(entry.user_id)

            sv_ids_original_root = self.original_root_id_lookup_vec(entry.edges_failsafe)
            sv_ids_current_root = self.root_id_lookup_vec(entry.edges_failsafe)

            if entry.is_merge:
                if len(np.unique(sv_ids_original_root)) != 1:
                    is_relevant_list.append(True)
                else:
                    is_relevant_list.append(False)

                if np.all(sv_ids_current_root == self.root_id):
                    is_in_neuron_list.append(True)
                else:
                    is_in_neuron_list.append(False)

                before_root_ids, after_root_ids = \
                    self._before_after_root_ids(entry)

                if before_root_ids[0] in before_root_ids_list[-1]:
                    before_root_ids_list.append(before_root_ids)
                else:
                    before_root_ids_list.append(before_root_ids[::-1])

                if after_root_ids_list[-1][0] not in before_root_ids:
                    after_root_ids_list[-1] = after_root_ids_list[-1][::-1]

                after_root_ids_list.append(np.array([after_root_ids[0], 0]))
            else:
                if len(np.unique(sv_ids_current_root)) != 1:
                    is_relevant_list.append(True)
                else:
                    is_relevant_list.append(False)

                if np.any(sv_ids_current_root == self.root_id):
                    is_in_neuron_list.append(True)
                else:
                    is_in_neuron_list.append(False)

                before_root_ids, after_root_ids = \
                    self._before_after_root_ids(entry)

                before_root_ids_list.append(np.array([before_root_ids[0], 0]))

                if after_root_ids_list[-1][0] not in before_root_ids:
                    after_root_ids_list[-1] = after_root_ids_list[-1][::-1]

                if after_root_ids[1] == self.root_id:
                    after_root_ids_list.append(after_root_ids[::-1])
                else:
                    after_root_ids_list.append(after_root_ids)

        before_root_ids_list = np.array(before_root_ids_list)
        after_root_ids_list = np.array(after_root_ids_list)

        self._tabular_changelog = pd.DataFrame.from_dict(
            {"operation_id": np.sort(entry_ids),
             "timestamp": timestamp_list,
             "user_id": user_list,
             "before_root_id_1": before_root_ids_list[:, 0],
             "before_root_id_2": before_root_ids_list[:, 1],
             "after_root_id_1": after_root_ids_list[:, 0],
             "after_root_id_2": after_root_ids_list[:, 1],
             "is_merge": is_merge_list,
             "in_neuron": is_in_neuron_list,
             "is_relevant": is_relevant_list})


    def _before_after_root_ids(self, entry):
        before_root_ids = np.unique(
            self.cg.get_roots(entry.edges_failsafe,
                              time_stamp=entry.timestamp -
                                         datetime.timedelta(seconds=.01)))

        after_root_ids = np.unique(
            self.cg.get_roots(entry.edges_failsafe,
                              time_stamp=entry.timestamp +
                                         datetime.timedelta(seconds=.01)))

        assert np.sum(np.in1d(before_root_ids, after_root_ids)) == 0

        return before_root_ids, after_root_ids


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
    def log_type(self):
        return "merge" if self.is_merge else "split"

    @property
    def root_ids(self):
        return self.row[column_keys.OperationLogs.RootID]

    @property
    def edges_failsafe(self):
        try:
            return np.array(self.sink_source_ids)
        except:
            if self.is_merge:
                return np.array(self.added_edges).flatten()
            if not self.is_merge:
                return np.array(self.removed_edges).flatten()

    @property
    def sink_source_ids(self):
        return np.concatenate([self.row[column_keys.OperationLogs.SinkID],
                               self.row[column_keys.OperationLogs.SourceID]])

    @property
    def added_edges(self):
        if not self.is_merge:
            raise cg_exceptions.InternalServerError

        return self.row[column_keys.OperationLogs.AddedEdge]

    @property
    def removed_edges(self):
        if self.is_merge:
            raise cg_exceptions.InternalServerError

        return self.row[column_keys.OperationLogs.RemovedEdge]

    @property
    def coordinates(self):
        return np.array([self.row[column_keys.OperationLogs.SourceCoordinate],
                         self.row[column_keys.OperationLogs.SinkCoordinate]])

    def __str__(self):
        return f"{self.user_id},{self.log_type},{self.root_ids},{self.timestamp}"
        
    def __iter__(self):
        attrs = [self.user_id, self.log_type, self.root_ids, self.timestamp]
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