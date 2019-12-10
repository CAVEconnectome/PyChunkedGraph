import os
import sys
import time
import datetime
import logging
from itertools import chain
from itertools import product
from functools import reduce
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytz
from cloudvolume import CloudVolume
from multiwrapper import multiprocessing_utils as mu

from . import cutting
from . import operation
from . import attributes
from . import exceptions
from .client import base
from .client.bigtable import BigTableClient
from .meta import ChunkedGraphMeta
from .utils import basetypes
from .utils import id_helpers
from .utils import generic as misc_utils
from .edges import Edges
from .edges import utils as edge_utils
from .chunks import utils as chunk_utils
from .chunks import hierarchy as chunk_hierarchy
from ..ingest import IngestConfig
from ..io.edges import get_chunk_edges

# TODO this should be part of deployment
HOME = os.path.expanduser("~")
# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    HOME + "/.cloudvolume/secrets/google-secret.json"
)


class ChunkedGraph:
    def __init__(
        self, meta: ChunkedGraphMeta, logger: Optional[logging.Logger] = None,
    ) -> None:
        self._meta = meta
        # TODO create client based on type
        # for now, just create bigtable client
        bt_client = BigTableClient(self._meta)
        self._client = bt_client
        self._id_client = bt_client
        self._setup_logger(logger)

    @property
    def meta(self) -> ChunkedGraphMeta:
        return self._meta

    @property
    def cv(self) -> CloudVolume:
        return self.meta.cv

    @property
    def client(self) -> base.SimpleClient:
        return self._client

    @property
    def id_client(self) -> base.ClientWithIDGen:
        return self._id_client

    @property
    def root_chunk_id(self):
        return self.get_chunk_id(layer=int(self.meta.layer_count))

    def update_meta(self, meta: ChunkedGraphMeta):
        """Updates graph meta."""
        self.client.update_graph_meta(meta)

    def update_provenance(self, provenance: IngestConfig):
        """Updates information about how the graph was created."""
        self.client.update_graph_provenance(provenance)

    def range_read_chunk(
        self,
        chunk_id: basetypes.CHUNK_ID,
        properties: Optional[
            Union[Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        time_stamp: Optional[datetime.datetime] = None,
    ) -> Dict:
        layer = self.get_chunk_layer(chunk_id)
        max_segment_id = self.id_client.get_max_segment_id(chunk_id=chunk_id)
        if layer == 1:
            max_segment_id = self.get_segment_id_limit(chunk_id)

        return self.client.read_nodes(
            start_id=self.get_node_id(np.uint64(0), chunk_id=chunk_id),
            end_id=self.get_node_id(max_segment_id, chunk_id=chunk_id),
            end_id_inclusive=True,
            properties=properties,
            end_time=time_stamp,
            end_time_inclusive=True,
        )

    def get_atomic_id_from_coord(
        self, x: int, y: int, z: int, parent_id: np.uint64, n_tries: int = 5
    ) -> np.uint64:
        """Determines atomic id given a coordinate."""
        if self.get_chunk_layer(parent_id) == 1:
            return parent_id

        x = int(x / 2 ** self.meta.data_source.CV_MIP)
        y = int(y / 2 ** self.meta.data_source.CV_MIP)

        checked = []
        atomic_id = None
        root_id = self.get_root(parent_id)

        for i_try in range(n_tries):
            # Define block size -- increase by one each try
            x_l = x - (i_try - 1) ** 2
            y_l = y - (i_try - 1) ** 2
            z_l = z - (i_try - 1) ** 2

            x_h = x + 1 + (i_try - 1) ** 2
            y_h = y + 1 + (i_try - 1) ** 2
            z_h = z + 1 + (i_try - 1) ** 2

            x_l = 0 if x_l < 0 else x_l
            y_l = 0 if y_l < 0 else y_l
            z_l = 0 if z_l < 0 else z_l

            # Get atomic ids from cloudvolume
            atomic_id_block = self.cv[x_l:x_h, y_l:y_h, z_l:z_h]
            atomic_ids, atomic_id_count = np.unique(atomic_id_block, return_counts=True)

            # sort by frequency and discard those ids that have been checked
            # previously
            sorted_atomic_ids = atomic_ids[np.argsort(atomic_id_count)]
            sorted_atomic_ids = sorted_atomic_ids[~np.in1d(sorted_atomic_ids, checked)]

            # For each candidate id check whether its root id corresponds to the
            # given root id
            for candidate_atomic_id in sorted_atomic_ids:
                ass_root_id = self.get_root(candidate_atomic_id)
                if ass_root_id == root_id:
                    # atomic_id is not None will be our indicator that the
                    # search was successful
                    atomic_id = candidate_atomic_id
                    break
                else:
                    checked.append(candidate_atomic_id)
            if atomic_id is not None:
                break
        # Returns None if unsuccessful
        return atomic_id

    def read_first_log_row(self):
        """Returns first log row."""
        for operation_id in range(1, 100):
            log_row = self.read_log_row(np.uint64(operation_id))
            if len(log_row) > 0:
                return log_row
        return None

    def get_parents(
        self,
        node_ids: Sequence[np.uint64],
        get_only_relevant_parents: bool = True,
        time_stamp: Optional[datetime.datetime] = None,
    ):
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parent_rows = self.client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Parent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )

        if not parent_rows:
            return None

        if get_only_relevant_parents:
            return np.array([parent_rows[node_id][0].value for node_id in node_ids])

        parents = []
        for node_id in node_ids:
            parents.append([(p.value, p.timestamp) for p in parent_rows[node_id]])
        return parents

    def get_parent(
        self,
        node_id: np.uint64,
        get_only_relevant_parent: bool = True,
        time_stamp: Optional[datetime.datetime] = None,
    ) -> Union[List[Tuple[np.uint64, datetime.datetime]], np.uint64]:
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parents = self.client.read_node(
            node_id,
            properties=attributes.Hierarchy.Parent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )

        if not parents:
            return None

        if get_only_relevant_parent:
            return parents[0].value
        return [(p.value, p.timestamp) for p in parents]

    def get_children(
        self, node_id: Union[Iterable[np.uint64], np.uint64], flatten: bool = False
    ) -> Union[Dict[np.uint64, np.ndarray], np.ndarray]:
        """Returns children for the specified NodeID or NodeIDs
        :param node_id: The NodeID or NodeIDs for which to retrieve children
        :type node_id: Union[Iterable[np.uint64], np.uint64]
        :param flatten: If True, combine all children into a single array, else generate a map
            of input ``node_id`` to their respective children.
        :type flatten: bool, default is True
        :return: Children for each requested NodeID. The return type depends on the ``flatten``
            parameter.
        :rtype: Union[Dict[np.uint64, np.ndarray], np.ndarray]
        """
        if np.isscalar(node_id):
            children = self.client.read_node(
                node_id=node_id, properties=attributes.Hierarchy.Child
            )
            if not children:
                return np.empty(0, dtype=basetypes.NODE_ID)
            return children[0].value
        else:
            children = self.client.read_nodes(
                node_ids=node_id, properties=attributes.Hierarchy.Child
            )
            if flatten:
                if not children:
                    return np.empty(0, dtype=basetypes.NODE_ID)
                return np.concatenate([x[0].value for x in children.values()])
            return {
                x: children[x][0].value
                if x in children
                else np.empty(0, dtype=basetypes.NODE_ID)
                for x in node_id
            }

    def get_latest_roots(
        self,
        time_stamp: Optional[datetime.datetime] = misc_utils.get_max_time(),
        n_threads: int = 1,
    ) -> Sequence[np.uint64]:
        """Reads _all_ root ids."""
        pass
        # return misc.get_latest_roots(self, time_stamp=time_stamp, n_threads=n_threads)

    def get_delta_roots(
        self,
        time_stamp_start: datetime.datetime,
        time_stamp_end: Optional[datetime.datetime] = None,
        min_seg_id: int = 1,
        n_threads: int = 1,
    ) -> Sequence[np.uint64]:
        """ Returns root ids that have expired or have been created between two timestamps
        :param time_stamp_start: datetime.datetime
            starting timestamp to return deltas from
        :param time_stamp_end: datetime.datetime
            ending timestamp to return deltasfrom
        :param min_seg_id: int (default=1)
            only search from this seg_id and higher (note not a node_id.. use get_seg_id)
        :param n_threads: int (default=1)
            number of threads to use in performing search
        :return new_ids, expired_ids: np.arrays of np.uint64
            new_ids is an array of root_ids for roots that were created after time_stamp_start
            and are still current as of time_stamp_end.
            expired_ids is list of node_id's for roots the expired after time_stamp_start
            but before time_stamp_end.
        """
        pass
        # return misc.get_delta_roots(
        #     self,
        #     time_stamp_start=time_stamp_start,
        #     time_stamp_end=time_stamp_end,
        #     min_seg_id=min_seg_id,
        #     n_threads=n_threads,
        # )

    def get_roots(
        self,
        node_ids: Sequence[np.uint64],
        time_stamp: Optional[datetime.datetime] = None,
        stop_layer: int = None,
        n_tries: int = 1,
    ):
        """ Takes node ids and returns the associated agglomeration ids
        :param node_ids: list of uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        layer_mask = np.ones(len(node_ids), dtype=np.bool)

        for _ in range(n_tries):
            layer_mask[self.get_chunk_layers(node_ids) >= stop_layer] = False
            parent_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
            for _ in range(int(stop_layer + 1)):
                filtered_ids = parent_ids[layer_mask]
                unique_ids, inverse = np.unique(filtered_ids, return_inverse=True)
                temp_ids = self.get_parents(unique_ids, time_stamp=time_stamp)
                if temp_ids is None:
                    break
                else:
                    parent_ids[layer_mask] = temp_ids[inverse]
                    layer_mask[self.get_chunk_layers(parent_ids) >= stop_layer] = False
                    if not np.any(self.get_chunk_layers(parent_ids) < stop_layer):
                        return parent_ids
            if not np.any(self.get_chunk_layers(parent_ids) < stop_layer):
                return parent_ids
            else:
                time.sleep(0.5)
        return parent_ids

    def get_root(
        self,
        node_id: np.uint64,
        time_stamp: Optional[datetime.datetime] = None,
        get_all_parents=False,
        stop_layer: int = None,
        n_tries: int = 1,
    ) -> Union[List[np.uint64], np.uint64]:
        """ Takes a node id and returns the associated agglomeration ids
        :param node_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parent_id = node_id
        all_parent_ids = []
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer

        for _ in range(n_tries):
            parent_id = node_id
            for _ in range(self.get_chunk_layer(node_id), int(stop_layer + 1)):
                temp_parent_id = self.get_parent(parent_id, time_stamp=time_stamp)
                if temp_parent_id is None:
                    break
                else:
                    parent_id = temp_parent_id
                    all_parent_ids.append(parent_id)
                    if self.get_chunk_layer(parent_id) >= stop_layer:
                        break
            if self.get_chunk_layer(parent_id) >= stop_layer:
                break
            else:
                time.sleep(0.5)

        if self.get_chunk_layer(parent_id) < stop_layer:
            raise Exception("Cannot find root id {}, {}".format(node_id, time_stamp))

        if get_all_parents:
            return np.array(all_parent_ids)
        else:
            return parent_id

    def get_all_parents_dict(
        self, node_id: basetypes.NODE_ID, time_stamp: Optional[datetime.datetime] = None
    ) -> dict:
        """Takes a node id and returns all parents up to root."""
        parent_ids = self.get_root(
            node_id=node_id, time_stamp=time_stamp, get_all_parents=True
        )
        return dict(zip(self.get_chunk_layers(parent_ids), parent_ids))

    # def read_consolidated_lock_timestamp
    # def read_lock_timestamp

    def get_latest_root_id(self, root_id: np.uint64) -> np.ndarray:
        """Returns the latest root id associated with the provided root id
        :param root_id: uint64
        :return: list of uint64s
        """
        id_working_set = [root_id]
        column = attributes.Hierarchy.NewParent
        latest_root_ids = []
        while len(id_working_set) > 0:
            next_id = id_working_set[0]
            del id_working_set[0]
            node = self.client.read_node(next_id, properties=column)
            # Check if a new root id was attached to this root id
            if node:
                id_working_set.extend(node[0].value)
            else:
                latest_root_ids.append(next_id)

        return np.unique(latest_root_ids)

    def get_future_root_ids(
        self,
        root_id: basetypes.NODE_ID,
        time_stamp: Optional[datetime.datetime] = misc_utils.get_max_time(),
    ) -> np.ndarray:
        """ Returns all future root ids emerging from this root
        This search happens in a monotic fashion. At no point are past root
        ids of future root ids taken into account.
        :param root_id: np.uint64
        :param time_stamp: None or datetime
            restrict search to ids created before this time_stamp
            None=search whole future
        :return: array of uint64
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        id_history = []
        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []
            for next_id in next_ids:
                row = self.client.read_node(
                    next_id,
                    properties=[
                        attributes.Hierarchy.NewParent,
                        attributes.Hierarchy.Child,
                    ],
                )
                if attributes.Hierarchy.NewParent in row:
                    ids = row[attributes.Hierarchy.NewParent][0].value
                    row_time_stamp = row[attributes.Hierarchy.NewParent][0].timestamp
                elif attributes.Hierarchy.Child in row:
                    ids = None
                    row_time_stamp = row[attributes.Hierarchy.Child][0].timestamp
                else:
                    raise exceptions.ChunkedGraphError(
                        "Error retrieving future root ID of %s" % next_id
                    )

                if row_time_stamp < time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)
                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids
        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_past_root_ids(
        self,
        root_id: np.uint64,
        time_stamp: Optional[datetime.datetime] = misc_utils.get_min_time(),
    ) -> np.ndarray:
        """ Returns all past root ids emerging from this root
        This search happens in a monotic fashion. At no point are future root
        ids of past root ids taken into account.
        :param root_id: np.uint64
        :param time_stamp: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole future
        :return: array of uint64
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        id_history = []
        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []
            for next_id in next_ids:
                row = self.client.read_node(
                    next_id,
                    properties=[
                        attributes.Hierarchy.FormerParent,
                        attributes.Hierarchy.Child,
                    ],
                )
                if attributes.Hierarchy.FormerParent in row:
                    ids = row[attributes.Hierarchy.FormerParent][0].value
                    row_time_stamp = row[attributes.Hierarchy.FormerParent][0].timestamp
                elif attributes.Hierarchy.Child in row:
                    ids = None
                    row_time_stamp = row[attributes.Hierarchy.Child][0].timestamp
                else:
                    raise exceptions.ChunkedGraphError(
                        "Error retrieving past root ID of %s" % next_id
                    )

                if row_time_stamp > time_stamp:
                    if ids is not None:
                        temp_next_ids.extend(ids)

                    if next_id != root_id:
                        id_history.append(next_id)

            next_ids = temp_next_ids
        return np.unique(np.array(id_history, dtype=np.uint64))

    def get_change_log(
        self,
        root_id: np.uint64,
        correct_for_wrong_coord_type: bool = True,
        time_stamp_past: Optional[datetime.datetime] = misc_utils.get_min_time(),
    ) -> dict:
        """ Returns all past root ids for this root
        This search happens in a monotic fashion. At no point are future root
        ids of past root ids taken into account.
        :param root_id: np.uint64
        :param correct_for_wrong_coord_type: bool
            pinky100? --> True
        :param time_stamp_past: None or datetime
            restrict search to ids created after this time_stamp
            None=search whole past
        :return: past ids, merge sv ids, merge edge coords, split sv ids
        """
        if time_stamp_past.tzinfo is None:
            time_stamp_past = pytz.UTC.localize(time_stamp_past)

        id_history = []
        merge_history = []
        merge_history_edges = []
        split_history = []
        next_ids = [root_id]
        while len(next_ids):
            temp_next_ids = []
            former_parent_col = attributes.Hierarchy.FormerParent
            nodes_d = self.client.read_nodes(
                node_ids=next_ids, properties=[former_parent_col]
            )
            for node in nodes_d.values():
                if attributes.Hierarchy.FormerParent in node:
                    if time_stamp_past > node[former_parent_col][0].timestamp:
                        continue
                    ids = node[former_parent_col][0].value
                    lock_col = attributes.Concurrency.Lock
                    former_node = self.client.read_node(ids[0], properties=[lock_col])
                    operation_id = former_node[lock_col][0].value
                    log_row = self.read_log_row(operation_id)
                    is_merge = attributes.OperationLogs.AddedEdge in log_row
                    for id_ in ids:
                        if id_ in id_history:
                            continue
                        id_history.append(id_)
                        temp_next_ids.append(id_)

                    if is_merge:
                        added_edges = log_row[attributes.OperationLogs.AddedEdge]
                        merge_history.append(added_edges)
                        coords = [
                            log_row[attributes.OperationLogs.SourceCoordinate],
                            log_row[attributes.OperationLogs.SinkCoordinate],
                        ]
                        if correct_for_wrong_coord_type:
                            # A little hack because we got the datatype wrong...
                            coords = [
                                np.frombuffer(coords[0]),
                                np.frombuffer(coords[1]),
                            ]
                            coords *= np.array(self.cv.scale["resolution"])
                        merge_history_edges.append(coords)
                    if not is_merge:
                        removed_edges = log_row[attributes.OperationLogs.RemovedEdge]
                        split_history.append(removed_edges)
                else:
                    continue
            next_ids = temp_next_ids
        return {
            "past_ids": np.unique(np.array(id_history, dtype=np.uint64)),
            "merge_edges": np.array(merge_history),
            "merge_edge_coords": np.array(merge_history_edges),
            "split_edges": np.array(split_history),
        }

    def _get_subgraph_higher_layer_nodes(
        self,
        node_id: basetypes.NODE_ID,
        bounding_box: Optional[Sequence[Sequence[int]]],
        return_layers: Sequence[int],
    ):
        def _get_subgraph_higher_layer_nodes_threaded(
            node_ids: Iterable[np.uint64],
        ) -> List[np.uint64]:
            children = self.get_children(node_ids, flatten=True)
            if len(children) > 0 and bounding_box is not None:
                chunk_coords = np.array(
                    [self.get_chunk_coordinates(c) for c in children]
                )
                child_layers = self.get_chunk_layers(children) - 2
                child_layers[child_layers < 0] = 0
                fanout = self.meta.graph_config.FANOUT
                bbox_layer = (
                    bounding_box[None] / (fanout ** child_layers)[:, None, None]
                )
                bound_check = np.array(
                    [
                        np.all(chunk_coords < bbox_layer[:, 1], axis=1),
                        np.all(chunk_coords + 1 > bbox_layer[:, 0], axis=1),
                    ]
                ).T
                bound_check_mask = np.all(bound_check, axis=1)
                children = children[bound_check_mask]
            return children

        if bounding_box is not None:
            bounding_box = np.array(bounding_box)

        layer = self.get_chunk_layer(node_id)
        assert layer > 1

        nodes_per_layer = {}
        child_ids = np.array([node_id], dtype=np.uint64)
        stop_layer = max(2, np.min(return_layers))

        if layer in return_layers:
            nodes_per_layer[layer] = child_ids

        while layer > stop_layer:
            # Use heuristic to guess the optimal number of threads
            child_id_layers = self.get_chunk_layers(child_ids)
            this_layer_m = child_id_layers == layer
            this_layer_child_ids = child_ids[this_layer_m]
            next_layer_child_ids = child_ids[~this_layer_m]

            n_child_ids = len(child_ids)
            this_n_threads = np.min([int(n_child_ids // 50000) + 1, mu.n_cpus])

            child_ids = np.fromiter(
                chain.from_iterable(
                    mu.multithread_func(
                        _get_subgraph_higher_layer_nodes_threaded,
                        np.array_split(this_layer_child_ids, this_n_threads),
                        n_threads=this_n_threads,
                        debug=this_n_threads == 1,
                    )
                ),
                np.uint64,
            )
            child_ids = np.concatenate([child_ids, next_layer_child_ids])
            layer -= 1
            if layer in return_layers:
                nodes_per_layer[layer] = child_ids
        return nodes_per_layer

    def get_subgraph_edges(
        self,
        agglomeration_id: np.uint64,
        bounding_box: Optional[Sequence[Sequence[int]]] = None,
        bb_is_coordinate: bool = False,
        connected_edges=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Return all atomic edges between supervoxels belonging to the 
        specified agglomeration ID within the defined bounding box
        """
        return self.get_subgraph(
            np.array([agglomeration_id]),
            bbox=bounding_box,
            bbox_is_coordinate=bb_is_coordinate,
        )

    def get_subgraph(
        self,
        agglomeration_ids: np.ndarray,
        bbox: Optional[Sequence[Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
        n_threads: int = 1,
        active_edges: bool = True,
        timestamp: datetime.datetime = None,
    ) -> Tuple[Dict, Dict]:
        """
        1. get level 2 children ids belonging to the agglomerations
        2. get relevant chunk ids from level 2 ids
        3. read edges from cloud storage (include fake edges from big table)
        4. get supervoxel ids from level 2 ids
        5. filter the edges with supervoxel ids
        6. optionally for each edge (v1,v2) active
           if parent(v1) == parent(v2) inactive otherwise
        7. returns dict of Agglomerations
        """
        level2_ids = []
        for agglomeration_id in agglomeration_ids:
            layer_nodes_d = self._get_subgraph_higher_layer_nodes(
                node_id=agglomeration_id,
                bounding_box=chunk_utils.normalize_bounding_box(
                    self.meta, bbox, bbox_is_coordinate
                ),
                return_layers=[2],
            )
            level2_ids.append(layer_nodes_d[2])
        level2_ids = np.concatenate(level2_ids)

        chunk_ids = self.get_chunk_ids_from_node_ids(level2_ids)
        chunk_edge_dicts = mu.multithread_func(
            self.read_chunk_edges,
            np.array_split(np.unique(chunk_ids), n_threads),
            n_threads=n_threads,
            debug=False,
        )
        edges_dict = edge_utils.concatenate_chunk_edges(chunk_edge_dicts)
        edges = reduce(lambda x, y: x + y, edges_dict.values())
        # # include fake edges
        # chunk_fake_edges_d = self.read_node_id_rows(
        #     node_ids=chunk_ids,
        #     columns=attributes.Connectivity.FakeEdges)
        # fake_edges = np.concatenate([list(chunk_fake_edges_d.values())])
        # if fake_edges.size:
        #     fake_edges = Edges(fake_edges[:,0], fake_edges[:,1])
        #     edges += fake_edges

        # group nodes and edges based on level 2 ids
        l2id_agglomeration_d = {}
        l2id_children_d = self.get_children(level2_ids)
        for l2id in l2id_children_d:
            supervoxels = l2id_children_d[l2id]
            filtered_edges = edge_utils.filter_edges(l2id_children_d[l2id], edges)
            if active_edges:
                filtered_edges = edge_utils.get_active_edges(
                    filtered_edges, l2id_children_d
                )
            # l2id_agglomeration_d[l2id] = Agglomeration(supervoxels, filtered_edges)
        return l2id_agglomeration_d

    def get_subgraph_nodes(
        self,
        agglomeration_id: np.uint64,
        bounding_box: Optional[Sequence[Sequence[int]]] = None,
        bb_is_coordinate: bool = False,
        return_layers: List[int] = [1],
    ) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """ Return all nodes belonging to the specified agglomeration ID within
            the defined bounding box and requested layers.
        :param agglomeration_id: np.uint64
        :param bounding_box: [[x_l, y_l, z_l], [x_h, y_h, z_h]]
        :param bb_is_coordinate: bool
        :param return_layers: List[int]
        :return: np.array of atomic IDs if single layer is requested,
                 Dict[int, np.array] if multiple layers are requested
        """

        def _get_subgraph_layer2_nodes(node_ids: Iterable[np.uint64]) -> np.ndarray:
            return self.get_children(node_ids, flatten=True)

        stop_layer = np.min(return_layers)
        bounding_box = chunk_utils.normalize_bounding_box(
            self.meta, bounding_box, bb_is_coordinate
        )

        # Layer 3+
        if stop_layer >= 2:
            nodes_per_layer = self._get_subgraph_higher_layer_nodes(
                node_id=agglomeration_id,
                bounding_box=bounding_box,
                return_layers=return_layers,
            )
        else:
            # Need to retrieve layer 2 even if the user doesn't require it
            nodes_per_layer = self._get_subgraph_higher_layer_nodes(
                node_id=agglomeration_id,
                bounding_box=bounding_box,
                return_layers=return_layers + [2],
            )

            # Layer 2
            child_ids = nodes_per_layer[2]
            if 2 not in return_layers:
                del nodes_per_layer[2]

            # Use heuristic to guess the optimal number of threads
            n_child_ids = len(child_ids)
            this_n_threads = np.min([int(n_child_ids // 50000) + 1, mu.n_cpus])

            child_ids = np.fromiter(
                chain.from_iterable(
                    mu.multithread_func(
                        _get_subgraph_layer2_nodes,
                        np.array_split(child_ids, this_n_threads),
                        n_threads=this_n_threads,
                        debug=this_n_threads == 1,
                    )
                ),
                dtype=np.uint64,
            )
            nodes_per_layer[1] = child_ids
        if len(nodes_per_layer) == 1:
            return list(nodes_per_layer.values())[0]
        else:
            return nodes_per_layer

    def add_edges(
        self,
        user_id: str,
        atomic_edges: Sequence[np.uint64],
        affinities: Sequence[np.float32] = None,
        source_coord: Sequence[int] = None,
        sink_coord: Sequence[int] = None,
        n_tries: int = 60,
    ) -> operation.GraphEditOperation.Result:
        """ Adds an edge to the chunkedgraph
            Multi-user safe through locking of the root node
            This function acquires a lock and ensures that it still owns the
            lock before executing the write.
        :param user_id: str
            unique id - do not just make something up, use the same id for the
            same user every time
        :param atomic_edges: list of two uint64s
            have to be from the same two root ids!
        :param affinities: list of np.float32 or None
            will eventually be set to 1 if None
        :param source_coord: list of int (n x 3)
        :param sink_coord: list of int (n x 3)
        :param n_tries: int
        :return: GraphEditOperation.Result
        """
        return operation.MergeOperation(
            self,
            user_id=user_id,
            added_edges=atomic_edges,
            affinities=affinities,
            source_coords=source_coord,
            sink_coords=sink_coord,
        ).execute()

    def remove_edges(
        self,
        user_id: str,
        source_ids: Sequence[np.uint64] = None,
        sink_ids: Sequence[np.uint64] = None,
        source_coords: Sequence[Sequence[int]] = None,
        sink_coords: Sequence[Sequence[int]] = None,
        atomic_edges: Sequence[Tuple[np.uint64, np.uint64]] = None,
        mincut: bool = True,
        bb_offset: Tuple[int, int, int] = (240, 240, 24),
        n_tries: int = 20,
    ) -> operation.GraphEditOperation.Result:
        """ Removes edges - either directly or after applying a mincut
            Multi-user safe through locking of the root node
            This function acquires a lock and ensures that it still owns the
            lock before executing the write.
        :param atomic_edges: list of 2 uint64
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :return: GraphEditOperation.Result
        """
        if mincut:
            return operation.MulticutOperation(
                self,
                user_id=user_id,
                source_ids=source_ids,
                sink_ids=sink_ids,
                source_coords=source_coords,
                sink_coords=sink_coords,
                bbox_offset=bb_offset,
            ).execute()

        if not atomic_edges:
            # Shim - can remove this check once all functions call the split properly/directly
            source_ids = [source_ids] if np.isscalar(source_ids) else source_ids
            sink_ids = [sink_ids] if np.isscalar(sink_ids) else sink_ids
            if len(source_ids) != len(sink_ids):
                raise exceptions.PreconditionError(
                    "Split operation require the same number of source and sink IDs"
                )
            atomic_edges = np.array([source_ids, sink_ids]).transpose()
        return operation.SplitOperation(
            self,
            user_id=user_id,
            removed_edges=atomic_edges,
            source_coords=source_coords,
            sink_coords=sink_coords,
        ).execute()

    def undo_operation(
        self, user_id: str, operation_id: np.uint64
    ) -> operation.GraphEditOperation.Result:
        """ Applies the inverse of a previous GraphEditOperation
        :param user_id: str
        :param operation_id: operation_id to be inverted
        :return: GraphEditOperation.Result
        """
        return operation.UndoOperation(
            self, user_id=user_id, operation_id=operation_id
        ).execute()

    def redo_operation(
        self, user_id: str, operation_id: np.uint64
    ) -> operation.GraphEditOperation.Result:
        """ Re-applies a previous GraphEditOperation
        :param user_id: str
        :param operation_id: operation_id to be repeated
        :return: GraphEditOperation.Result
        """
        return operation.RedoOperation(
            self, user_id=user_id, operation_id=operation_id
        ).execute()

    def _setup_logger(self, logger: Optional[logging.Logger] = None) -> None:
        if logger is None:
            self.logger = logging.getLogger(f"{self.meta.graph_config.ID}")
            self.logger.setLevel(logging.WARNING)
            if not self.logger.handlers:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(logging.WARNING)
                self.logger.addHandler(sh)
        else:
            self.logger = logger

    def _run_multicut(
        self,
        source_ids: Sequence[np.uint64],
        sink_ids: Sequence[np.uint64],
        source_coords: Sequence[Sequence[int]],
        sink_coords: Sequence[Sequence[int]],
        bb_offset: Tuple[int, int, int] = (120, 120, 12),
    ):
        time_start = time.time()
        bb_offset = np.array(list(bb_offset))
        source_coords = np.array(source_coords)
        sink_coords = np.array(sink_coords)

        # Decide a reasonable bounding box (NOT guaranteed to be successful!)
        coords = np.concatenate([source_coords, sink_coords])
        bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]

        bounding_box[0] -= bb_offset
        bounding_box[1] += bb_offset

        # Verify that sink and source are from the same root object
        root_ids = set()
        for source_id in source_ids:
            root_ids.add(self.get_root(source_id))
        for sink_id in sink_ids:
            root_ids.add(self.get_root(sink_id))

        if len(root_ids) > 1:
            raise exceptions.PreconditionError(
                f"All supervoxel must belong to the same object. Already split?"
            )

        self.logger.debug(
            "Get roots and check: %.3fms" % ((time.time() - time_start) * 1000)
        )
        time_start = time.time()  # ------------------------------------------

        root_id = root_ids.pop()

        # Get edges between local supervoxels
        chunk_size = self.meta.graph_config.CHUNK_SIZE
        n_chunks_affected = np.product(
            (np.ceil(bounding_box[1] / chunk_size)).astype(np.int)
            - (np.floor(bounding_box[0] / chunk_size)).astype(np.int)
        )

        self.logger.debug("Number of affected chunks: %d" % n_chunks_affected)
        self.logger.debug(f"Bounding box: {bounding_box}")
        self.logger.debug(f"Bounding box padding: {bb_offset}")
        self.logger.debug(f"Source ids: {source_ids}")
        self.logger.debug(f"Sink ids: {sink_ids}")
        self.logger.debug(f"Root id: {root_id}")

        edges, affs, _ = self.get_subgraph_edges(
            root_id, bounding_box=bounding_box, bb_is_coordinate=True
        )
        self.logger.debug(
            f"Get edges and affs: " f"{(time.time() - time_start) * 1000:.3f}ms"
        )

        time_start = time.time()  # ------------------------------------------

        if len(edges) == 0:
            raise exceptions.PreconditionError(
                f"No local edges found. " f"Something went wrong with the bounding box?"
            )

        # Compute mincut
        atomic_edges = cutting.mincut(edges, affs, source_ids, sink_ids)
        self.logger.debug(f"Mincut: {(time.time() - time_start) * 1000:.3f}ms")
        if len(atomic_edges) == 0:
            raise exceptions.PostconditionError(f"Mincut failed. Try again...")

        # # Check if any edge in the cutset is infinite (== between chunks)
        # # We would prevent such a cut
        #
        # atomic_edges_flattened_view = atomic_edges.view(dtype='u8,u8')
        # edges_flattened_view = edges.view(dtype='u8,u8')
        #
        # cutset_mask = np.in1d(edges_flattened_view, atomic_edges_flattened_view)
        #
        # if np.any(np.isinf(affs[cutset_mask])):
        #     self.logger.error("inf in cutset")
        #     return False, None
        return atomic_edges

    # OPERATION LOGGING
    def read_logs(self, operation_ids: Optional[List[np.uint64]] = None):
        if not operation_ids:
            log_records_d = self.client.read_nodes(
                start_id=np.uint64(0),
                end_id=self.id_client.get_max_operation_id(),
                end_id_inclusive=True,
                properties=attributes.OperationLogs.all(),
            )
        else:
            log_records_d = self.client.read_nodes(
                node_ids=operation_ids, properties=attributes.OperationLogs.all()
            )

        if len(log_records_d) == 0:
            return {}

        for operation_id in log_records_d:
            log_record = log_records_d[operation_id]
            timestamp = log_record[attributes.OperationLogs.RootID][0].timestamp
            log_record.update((column, v[0].value) for column, v in log_record.items())
            log_record["timestamp"] = timestamp
        return log_records_d

    # HELPERS
    def get_node_id(
        self,
        segment_id: np.uint64,
        chunk_id: Optional[np.uint64] = None,
        layer: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
    ) -> np.uint64:
        return id_helpers.get_node_id(
            self.meta, segment_id, chunk_id=chunk_id, layer=layer, x=x, y=y, z=z
        )

    def get_segment_id(self, node_id: basetypes.NODE_ID):
        return id_helpers.get_segment_id(self.meta, node_id)

    def get_segment_id_limit(self, node_or_chunk_id: basetypes.NODE_ID):
        return id_helpers.get_segment_id_limit(self.meta, node_or_chunk_id)

    def get_chunk_layer(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_utils.get_chunk_layer(self.meta, node_or_chunk_id)

    def get_chunk_layers(self, node_or_chunk_ids: Sequence):
        return chunk_utils.get_chunk_layers(self.meta, node_or_chunk_ids)

    def get_chunk_coordinates(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_utils.get_chunk_coordinates(self.meta, node_or_chunk_id)

    def get_chunk_id(
        self,
        node_id: basetypes.NODE_ID = None,
        layer: Optional[int] = None,
        x: Optional[int] = 0,
        y: Optional[int] = 0,
        z: Optional[int] = 0,
    ):
        return chunk_utils.get_chunk_id(
            self.meta, node_id=node_id, layer=layer, x=x, y=y, z=z
        )

    def get_chunk_ids_from_node_ids(self, node_ids: Sequence):
        return chunk_utils.get_chunk_ids_from_node_ids(self.meta, node_ids)

    def get_children_chunk_ids(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_children_chunk_ids(self.meta, node_or_chunk_id)

    def get_parent_chunk_ids(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_parent_chunk_ids(self.meta, node_or_chunk_id)

    def get_parent_chunk_id_dict(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_parent_chunk_id_dict(self.meta, node_or_chunk_id)

    def get_cross_chunk_edges_layer(self, cross_edges: Iterable):
        return edge_utils.get_cross_chunk_edges_layer(self.meta, cross_edges)

    def read_chunk_edges(self, chunk_ids: Iterable, cv_threads: int = 1) -> dict:
        return get_chunk_edges(
            self.meta.data_source.EDGES,
            [self.get_chunk_coordinates(chunk_id) for chunk_id in chunk_ids],
            cv_threads=cv_threads,
        )
