import collections
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import fastremap
from networkx.algorithms.dag import ancestors as nx_ancestors

from .attributes import OperationLogs
from .utils import basetypes


class SegmentHistory:
    def __init__(
        self,
        cg,
        root_ids,
        timestamp_past: datetime = None,
        timestamp_future: datetime = None,
    ):
        self.cg = cg
        if isinstance(root_ids, Iterable):
            self.root_ids = np.array(root_ids)
        else:
            self.root_ids = np.array([root_ids])

        for root_id in self.root_ids:
            assert cg.is_root(root_id), f"{root_id} is no root"

        self.timestamp_past = cg.get_earliest_timestamp()
        if timestamp_past is not None:
            self.timestamp_past = timestamp_past

        self.timestamp_future = datetime.now(timezone.utc)
        if timestamp_future is None:
            self.timestamp_future = timestamp_future

        self._lineage_graph = None
        self._operation_id_root_id_dict = None
        self._root_id_operation_id_dict = None
        self._root_id_timestamp_dict = None
        self._log_rows_cache = None
        self._tabular_changelogs = None

    @property
    def lineage_graph(self):
        from .lineage import lineage_graph

        if self._lineage_graph is None:
            self._lineage_graph = lineage_graph(
                self.cg, self.root_ids, self.timestamp_past, self.timestamp_future
            )
        return self._lineage_graph

    @property
    def root_id_operation_id_dict(self):
        if self._root_id_operation_id_dict is None:
            self._root_id_operation_id_dict = dict(
                self.lineage_graph.nodes.data("operation_id", default=0)
            )
        return self._root_id_operation_id_dict

    @property
    def root_id_timestamp_dict(self):
        if self._root_id_timestamp_dict is None:
            self._root_id_timestamp_dict = dict(
                self.lineage_graph.nodes.data("timestamp", default=0)
            )
        return self._root_id_timestamp_dict

    @property
    def operation_id_root_id_dict(self):
        if self._operation_id_root_id_dict is None:
            self._operation_id_root_id_dict = collections.defaultdict(list)
            for root_id, operation_id in self.root_id_operation_id_dict.items():
                self._operation_id_root_id_dict[operation_id].append(root_id)
        return self._operation_id_root_id_dict

    @property
    def operation_ids(self):
        return np.array(
            list(self.operation_id_root_id_dict.keys()), dtype=basetypes.OPERATION_ID
        )

    @property
    def _log_rows(self):
        if self._log_rows_cache is None:
            self._log_rows_cache = self.cg.client.read_log_entries(self.operation_ids)
        return self._log_rows_cache

    @property
    def tabular_changelogs(self):
        if self._tabular_changelogs is None:
            self._tabular_changelogs = self._build_tabular_changelogs()
        return self._tabular_changelogs

    @property
    def tabular_changelogs_filtered(self):
        filtered_tabular_changelogs = {}
        for root_id in self.root_ids:
            filtered_tabular_changelogs[root_id] = self.tabular_changelog(
                root_id=root_id, filtered=True
            )
        return filtered_tabular_changelogs

    def collect_edited_sv_ids(self, root_id=None):
        if root_id is None:
            operation_ids = self.past_operation_ids()
        else:
            assert root_id in self.root_ids
            operation_ids = self.past_operation_ids(root_id=root_id)

        edited_sv_ids = []
        for operation_id in operation_ids:
            edited_sv_ids.extend(self.log_entry(operation_id).edges_failsafe)

        if len(edited_sv_ids) > 0:
            return fastremap.unique(np.array(edited_sv_ids))
        else:
            return np.empty((0), dtype=np.uint64)

    def _build_tabular_changelogs(self):
        from pandas import DataFrame

        tabular_changelogs = {}
        all_user_ids = []
        root_lookup = lambda sv_ids, ts: dict(
            zip(
                sv_ids,
                self.cg.get_roots(sv_ids, time_stamp=ts),
            )
        ).get

        earliest_ts = self.cg.get_earliest_timestamp()
        root_ts_d = dict(
            zip(
                self.root_ids,
                self.cg.get_node_timestamps(self.root_ids, return_numpy=False),
            )
        )
        for root_id in self.root_ids:
            root_ts = root_ts_d[root_id]
            edited_sv_ids = self.collect_edited_sv_ids(root_id=root_id)
            current_root_id_lookup_vec = np.vectorize(
                root_lookup(edited_sv_ids, root_ts)
            )
            original_root_id_lookup_vec = np.vectorize(
                root_lookup(edited_sv_ids, earliest_ts)
            )

            is_merge_list = []
            is_in_neuron_list = []
            is_relevant_list = []
            timestamp_list = []
            user_list = []
            before_root_ids_list = []
            after_root_ids_list = []

            operation_ids = self.past_operation_ids(root_id=root_id)
            sorted_operation_ids = np.sort(operation_ids)
            for operation_id in sorted_operation_ids:
                entry = self.log_entry(operation_id)
                is_merge_list.append(entry.is_merge)
                timestamp_list.append(entry.timestamp)
                user_list.append(entry.user_id)

                sv_ids_original_root = original_root_id_lookup_vec(entry.edges_failsafe)
                sv_ids_current_root = current_root_id_lookup_vec(entry.edges_failsafe)
                before_ids = list(self.operation_id_root_id_dict[operation_id])
                after_root_ids_list.append(
                    list(self.lineage_graph.neighbors(before_ids[0]))
                )
                before_root_ids_list.append(before_ids)
                if entry.is_merge:
                    is_relevant_list.append(len(np.unique(sv_ids_original_root)) != 1)
                    is_in_neuron_list.append(np.all(sv_ids_current_root == root_id))
                else:
                    is_relevant_list.append(len(np.unique(sv_ids_current_root)) != 1)
                    is_in_neuron_list.append(np.any(sv_ids_current_root == root_id))

            all_user_ids.extend(user_list)
            tabular_changelogs[root_id] = DataFrame.from_dict(
                {
                    "operation_id": sorted_operation_ids,
                    "timestamp": timestamp_list,
                    "user_id": user_list,
                    "before_root_ids": before_root_ids_list,
                    "after_root_ids": after_root_ids_list,
                    "is_merge": is_merge_list,
                    "in_neuron": is_in_neuron_list,
                    "is_relevant": is_relevant_list,
                }
            )
        return tabular_changelogs

    def log_entry(self, operation_id):
        ts = self._log_rows[operation_id]["timestamp"]
        return LogEntry(self._log_rows[operation_id], timestamp=ts)

    def change_log_summary(self, root_id=None, filtered=False):
        if root_id is None:
            root_ids = self.root_ids
        else:
            assert root_id in self.root_ids
            root_ids = [root_id]

        for root_id in root_ids:
            tabular_changelog = self.tabular_changelog(root_id, filtered=filtered)

            user_ids = np.array(tabular_changelog[["user_id"]]).reshape(-1)
            u_user_ids = np.unique(user_ids)

            n_splits = 0
            n_mergers = 0
            n_edits = 0
            user_dict = collections.defaultdict(collections.Counter)
            for user_id in u_user_ids:
                m = user_ids == user_id
                n_user_edits = np.sum(m)
                n_user_mergers = int(np.sum(tabular_changelog[["is_merge"]][m]))
                n_user_splits = n_user_edits - n_user_mergers

                user_dict[user_id]["n_splits"] += n_user_splits
                user_dict[user_id]["n_mergers"] += n_user_mergers
                n_splits += n_user_splits
                n_mergers += n_user_mergers
                n_edits += n_user_edits

            before_col = list(tabular_changelog["before_root_ids"])
            if len(before_col) == 0:
                past_ids = np.empty((0), dtype=basetypes.NODE_ID)
            else:
                past_ids = np.concatenate(before_col, dtype=basetypes.NODE_ID)

            operation_ids = np.array(
                tabular_changelog["operation_id"], dtype=basetypes.NODE_ID
            )

        return {
            "n_splits": n_splits,
            "n_mergers": n_mergers,
            "user_info": user_dict,
            "operations_ids": operation_ids,
            "past_ids": past_ids,
        }

    def merge_log(self, root_id=None, correct_for_wrong_coord_type=True):
        if root_id is None:
            root_ids = self.root_ids
        else:
            assert root_id in self.root_ids
            root_ids = [root_id]

        added_edges = []
        added_edge_coords = []

        for root_id in root_ids:
            for operation_id in self.past_operation_ids(root_id=root_id):
                log_entry = self.log_entry(operation_id)
                if not log_entry.is_merge:
                    continue

                added_edges.append(log_entry.added_edges)
                coords = log_entry.coordinates
                if correct_for_wrong_coord_type:
                    # A little hack because we got the datatype wrong...
                    coords = [np.frombuffer(coords[0]), np.frombuffer(coords[1])]
                    coords *= self.cg.meta.resolution
                added_edge_coords.append(coords)
        return {"merge_edges": added_edges, "merge_edge_coords": added_edge_coords}

    def past_operation_ids(self, root_id=None):
        if root_id is None:
            root_ids = self.root_ids
        else:
            assert root_id in self.root_ids
            root_ids = [root_id]

        ancs = []
        for root_id in root_ids:
            ancs.extend(nx_ancestors(self.lineage_graph, root_id))

        if len(ancs) == 0:
            return np.array([], dtype=int)

        ancs = fastremap.unique(np.array(ancs, dtype=np.uint64))
        operation_ids = []
        for anc in ancs:
            operation_ids.append(self.root_id_operation_id_dict.get(anc, 0))

        operation_ids = np.array(operation_ids)
        operation_ids = fastremap.unique(operation_ids)
        operation_ids = operation_ids[operation_ids != 0]
        return operation_ids

    def tabular_changelog(self, root_id=None, filtered=False):
        if len(self.root_ids) == 1:
            root_id = self.root_ids[0]
        else:
            assert root_id is not None

        tabular_changelog = self.tabular_changelogs[root_id].copy()
        if filtered:
            in_neuron = np.array(tabular_changelog[["in_neuron"]])
            is_relevant = np.array(tabular_changelog[["is_relevant"]])
            inclusion_mask = np.logical_and(in_neuron, is_relevant).reshape(-1)

            tabular_changelog = tabular_changelog[inclusion_mask]
            tabular_changelog = tabular_changelog.drop("in_neuron", axis=1)
            tabular_changelog = tabular_changelog.drop("is_relevant", axis=1)
        return tabular_changelog

    def last_edit_timestamp(self, root_id=None):
        assert root_id in self.root_ids
        return self.root_id_timestamp_dict[root_id]

    def past_future_id_mapping(self, root_id=None):
        from networkx.algorithms.dag import descendants as nx_descendants

        if root_id is None:
            root_ids = self.root_ids
        else:
            assert root_id in self.root_ids
            root_ids = [root_id]

        in_degree_dict = dict(self.lineage_graph.in_degree)
        out_degree_dict = dict(self.lineage_graph.out_degree)
        in_degree_dict_vec = np.vectorize(in_degree_dict.get)
        out_degree_dict_vec = np.vectorize(out_degree_dict.get)
        past_id_mapping = {}
        future_id_mapping = {}
        for root_id in root_ids:
            ancestors = np.array(
                list(nx_ancestors(self.lineage_graph, root_id)), dtype=np.uint64
            )
            if len(ancestors) == 0:
                past_id_mapping[int(root_id)] = [root_id]
            else:
                anc_in_degrees = in_degree_dict_vec(ancestors)
                past_id_mapping[int(root_id)] = ancestors[anc_in_degrees == 0]

        past_ids = fastremap.unique(np.concatenate(list(past_id_mapping.values())))
        for past_id in past_ids:
            descendants = np.array(
                list(nx_descendants(self.lineage_graph, past_id)) + [past_id],
                dtype=np.uint64,
            )
            if len(descendants) == 0:
                future_id_mapping[past_id] = past_id
                continue

            out_degrees = out_degree_dict_vec(descendants)
            if 2 in out_degrees or np.sum(out_degrees == 0) > 1:
                continue

            single_degree_descendants = descendants[
                out_degree_dict_vec(descendants) == 1
            ]
            if len(single_degree_descendants) == 0:
                future_id_mapping[past_id] = descendants[out_degrees == 0][0]
                continue

            partner_in_degrees = in_degree_dict_vec(
                [
                    list(self.lineage_graph.neighbors(d))[0]
                    for d in single_degree_descendants
                ]
            )
            if 1 in partner_in_degrees:
                continue
            future_id_mapping[past_id] = descendants[out_degrees == 0][0]
        return past_id_mapping, future_id_mapping


class LogEntry:
    def __init__(self, row, timestamp):
        self.row = row
        self.timestamp = timestamp

    @property
    def is_merge(self):
        return OperationLogs.AddedEdge in self.row

    @property
    def user_id(self):
        return self.row[OperationLogs.UserID]

    @property
    def log_type(self):
        return "merge" if self.is_merge else "split"

    @property
    def root_ids(self):
        return self.row[OperationLogs.RootID]

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
        return np.concatenate(
            [
                self.row[OperationLogs.SinkID],
                self.row[OperationLogs.SourceID],
            ]
        )

    @property
    def added_edges(self):
        assert self.is_merge, "Not a merge operation."
        return self.row[OperationLogs.AddedEdge]

    @property
    def removed_edges(self):
        assert not self.is_merge, "Not a split operation."
        return self.row[OperationLogs.RemovedEdge]

    @property
    def coordinates(self):
        return np.array(
            [
                self.row[OperationLogs.SourceCoordinate],
                self.row[OperationLogs.SinkCoordinate],
            ]
        )

    def __iter__(self):
        attrs = [self.user_id, self.log_type, self.root_ids, self.timestamp]
        for attr in attrs:
            yield attr

    def __str__(self):
        return ",".join([str(x) for x in self])


def get_all_log_entries(cg):
    log_entries = []
    log_rows = cg.client.read_log_entries()
    for operation_id in range(cg.client.get_max_operation_id()):
        try:
            log_entries.append(
                LogEntry(log_rows[operation_id], log_rows[operation_id]["timestamp"])
            )
        except KeyError:
            continue
    return log_entries
