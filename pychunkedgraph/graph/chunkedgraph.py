import sys
import time
import typing
import datetime
import logging
from itertools import chain
from itertools import product
from functools import reduce
from collections import defaultdict

import numpy as np
import pytz
from cloudvolume import CloudVolume
from multiwrapper import multiprocessing_utils as mu

from . import types
from . import cutting
from . import operation
from . import attributes
from . import exceptions
from .client import base
from .client.bigtable import BigTableClient
from .meta import ChunkedGraphMeta
from .meta import BackendClientInfo
from .utils import basetypes
from .utils import id_helpers
from .utils import generic as misc_utils
from .utils.context_managers import TimeIt
from .edges import Edges
from .edges import utils as edge_utils
from .chunks import utils as chunk_utils
from .chunks import hierarchy as chunk_hierarchy
from ..ingest import IngestConfig
from ..io.edges import get_chunk_edges


# TODO logging with context manager?


class ChunkedGraph:
    def __init__(
        self,
        *,
        graph_id: str = None,
        meta: ChunkedGraphMeta = None,
        client_info: BackendClientInfo = BackendClientInfo(),
    ):
        """
        1. New graph
           Requires `meta`; if `client_info` is not passed the default client is used.
           After creating `ChunkedGraph` instance, run instance.create().
        2. Existing graph in default client
           Requires `graph_id`.
        3. Existing graphs in other projects/clients,
           Requires `graph_id` and `client_info`.
        """
        # TODO create client based on type
        # for now, just use BigTableClient

        if meta:
            graph_id = meta.graph_config.ID_PREFIX + meta.graph_config.ID
            bt_client = BigTableClient(graph_id, config=client_info.CONFIG)
            self._meta = meta
        else:
            bt_client = BigTableClient(graph_id, config=client_info.CONFIG)
            self._meta = bt_client.read_graph_meta()

        self._client = bt_client
        self._id_client = bt_client

    @property
    def meta(self) -> ChunkedGraphMeta:
        return self._meta

    @property
    def client(self) -> base.SimpleClient:
        return self._client

    @property
    def id_client(self) -> base.ClientWithIDGen:
        return self._id_client

    def create(self):
        """Creates the graph in storage client and stores meta."""
        self._client.create_graph(self._meta)

    def update_meta(self, meta: ChunkedGraphMeta):
        """Update meta of an already existing graph."""
        self.client.update_graph_meta(meta)

    def range_read_chunk(
        self,
        chunk_id: basetypes.CHUNK_ID,
        properties: typing.Optional[
            typing.Union[typing.Iterable[attributes._Attribute], attributes._Attribute]
        ] = None,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Dict:
        """Read all nodes in a chunk."""
        layer = self.get_chunk_layer(chunk_id)
        max_node_id = self.id_client.get_max_node_id(chunk_id=chunk_id)
        if layer == 1:
            max_node_id = chunk_id | self.get_segment_id_limit(chunk_id)

        return self.client.read_nodes(
            start_id=self.get_node_id(np.uint64(0), chunk_id=chunk_id),
            end_id=max_node_id,
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
            atomic_id_block = self.meta.cv[x_l:x_h, y_l:y_h, z_l:z_h]
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

    def get_parents(
        self,
        node_ids: typing.Sequence[np.uint64],
        current: bool = True,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ):
        """
        If current=True returns only the latest parents.
        Else all parents along with timestamps.
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parent_rows = self.client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Parent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )
        if not parent_rows:
            return None
        if current:
            return np.array([parent_rows[node_id][0].value for node_id in node_ids])
        parents = []
        for node_id in node_ids:
            parents.append([(p.value, p.timestamp) for p in parent_rows[node_id]])
        return parents

    def get_parent(
        self,
        node_id: np.uint64,
        get_only_relevant_parent: bool = True,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Union[typing.List[typing.Tuple], np.uint64]:
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
        self,
        node_id_or_ids: typing.Union[typing.Iterable[np.uint64], np.uint64],
        flatten: bool = False,
    ) -> typing.Union[typing.Dict, np.ndarray]:
        """
        Children for the specified NodeID or NodeIDs.
        If flatten == True, an array is returned, else a dict {node_id: children}.
        """
        if np.isscalar(node_id_or_ids):
            children = self.client.read_node(
                node_id=node_id_or_ids, properties=attributes.Hierarchy.Child
            )
            if not children:
                return np.empty(0, dtype=basetypes.NODE_ID)
            return children[0].value
        children = self.client.read_nodes(
            node_ids=node_id_or_ids, properties=attributes.Hierarchy.Child
        )
        if flatten:
            if not children:
                return np.empty(0, dtype=basetypes.NODE_ID)
            return np.concatenate([x[0].value for x in children.values()])
        return {
            x: children[x][0].value
            if x in children
            else np.empty(0, dtype=basetypes.NODE_ID)
            for x in node_id_or_ids
        }

    def get_atomic_cross_edges(
        self, node_ids: typing.Iterable
    ) -> typing.Dict[np.uint64, typing.Dict[int, typing.Iterable]]:
        """Returns cross edges for level 2 IDs."""
        properties = [
            attributes.Connectivity.CrossChunkEdge[l]
            for l in range(2, self.meta.layer_count)
        ]
        node_edges_d_d = self.client.read_nodes(
            node_ids=node_ids, properties=properties
        )

        result = {}
        for node_id, edges_d in node_edges_d_d.items():
            result[node_id] = {
                prop.index: val[0].value.copy() for prop, val in edges_d.items()
            }
        return result

    def get_cross_chunk_edges(
        self,
        node_ids: np.ndarray,
        *,
        nodes_cache: typing.Dict[np.uint64, types.Node] = None,
    ) -> typing.Dict[np.uint64, typing.Dict[int, typing.Iterable]]:
        """
        Cross chunk edges for `node_id` at `node_layer`.
        The edges are between node IDs at the `node_layer`, not atomic cross edges.
        Returns dict {layer_id: cross_edges}
            The first layer (>= `node_layer`) with atleast one cross chunk edge.
            For current use-cases, other layers are not relevant.

        For performance, only children that lie along chunk boundary are considered.
        Cross edges that belong to inner level 2 IDs are subsumed within the chunk.
        This is because cross edges are stored only in level 2 IDs.

        If `nodes_cache` is passed, IDs are first looked up in the cache.
        If the ID is not in the cache, it is read from storage.
        This is necessary when editing because the newly created IDs are 
        not yet written to storage. But it can also be used as cache.
        """
        result = {}
        if not node_ids.size:
            return result
        node_l2ids_d = self._get_bounding_l2_children(node_ids, cache=nodes_cache)
        all_children = np.fromiter(node_l2ids_d.values(), dtype=basetypes.NODE_ID)
        l2_edges_d_d = self.get_atomic_cross_edges(all_children)
        for node_id in node_ids:
            l2_edges_ds = [l2_edges_d_d[l2_id] for l2_id in node_l2ids_d[node_id]]
            result[node_id] = self.get_min_layer_cross_edges(node_id, l2_edges_ds)
        return result

    def get_min_layer_cross_edges(
        self, node_id: basetypes.NODE_ID, l2id_atomic_cross_edges_ds: typing.Iterable,
    ):
        """
        Find edges at relevant min_layer >= node_layer.
        `l2id_atomic_cross_edges_ds` is a list of atomic cross edges of
        level 2 IDs that are descendants of `node_id`.
        """
        node_layer = self.get_chunk_layer(node_id)
        min_layer = self.meta.layer_count
        for edges_d in l2id_atomic_cross_edges_ds:
            layer_, _ = edge_utils.filter_min_layer_cross_edges(
                self.meta, edges_d, node_layer=node_layer
            )
            min_layer = min(min_layer, layer_)

        edges = [types.empty_2d]
        for edges_d in l2id_atomic_cross_edges_ds:
            edges.append(edges_d.get(min_layer, types.empty_2d))
        edges = np.concatenate(edges)
        edges[:, 0] = self.get_root(node_id, stop_layer=min_layer)
        edges[:, 1] = self.get_roots(edges[:, 1], stop_layer=min_layer)
        return {min_layer: np.unique(edges, axis=0) if edges.size else types.empty_2d}

    def get_latest_roots(
        self,
        time_stamp: typing.Optional[datetime.datetime] = misc_utils.get_max_time(),
        n_threads: int = 1,
    ) -> typing.Sequence[np.uint64]:
        """Reads _all_ root ids."""
        pass
        # return misc.get_latest_roots(self, time_stamp=time_stamp, n_threads=n_threads)

    def get_roots(
        self,
        node_ids: typing.Sequence[np.uint64],
        time_stamp: typing.Optional[datetime.datetime] = None,
        stop_layer: int = None,
        n_tries: int = 1,
    ):
        """Takes node ids and returns the associated agglomeration ids."""
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        for _ in range(n_tries):
            layer_mask = self.get_chunk_layers(node_ids) < stop_layer
            parent_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
            for _ in range(int(stop_layer + 1)):
                filtered_ids = parent_ids[layer_mask]
                unique_ids, inverse = np.unique(filtered_ids, return_inverse=True)
                temp_ids = self.get_parents(unique_ids, time_stamp=time_stamp)
                if temp_ids is None:
                    break
                else:
                    parent_ids[layer_mask] = temp_ids[inverse]
                    if not np.any(self.get_chunk_layers(parent_ids) < stop_layer):
                        return parent_ids
                    layer_mask[self.get_chunk_layers(parent_ids) >= stop_layer] = False
            if not np.any(self.get_chunk_layers(parent_ids) < stop_layer):
                return parent_ids
            else:
                time.sleep(0.5)
        return parent_ids

    def get_root(
        self,
        node_id: np.uint64,
        time_stamp: typing.Optional[datetime.datetime] = None,
        get_all_parents=False,
        stop_layer: int = None,
        n_tries: int = 1,
    ) -> typing.Union[typing.List[np.uint64], np.uint64]:
        """ Takes a node id and returns the associated agglomeration ids
        :param node_id: uint64
        :param time_stamp: None or datetime
        :return: np.uint64
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parent_id = node_id
        all_parent_ids = []
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        if self.get_chunk_layer(parent_id) == stop_layer:
            return node_id

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
            raise Exception(
                f"Cannot find root id {node_id}, {stop_layer}, {time_stamp}"
            )

        if get_all_parents:
            return np.array(all_parent_ids)
        else:
            return parent_id

    def get_all_parents_dict(
        self,
        node_id: basetypes.NODE_ID,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> dict:
        """Takes a node id and returns all parents up to root."""
        parent_ids = self.get_root(
            node_id=node_id, time_stamp=time_stamp, get_all_parents=True
        )
        return dict(zip(self.get_chunk_layers(parent_ids), parent_ids))

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
        time_stamp: typing.Optional[datetime.datetime] = misc_utils.get_max_time(),
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
        time_stamp: typing.Optional[datetime.datetime] = misc_utils.get_min_time(),
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

    def get_subgraph(
        self,
        node_ids: np.ndarray,
        bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
        timestamp: datetime.datetime = None,
        layer_2: bool = False,
    ) -> typing.Dict:
        """
        1. get level 2 children ids belonging to the agglomerations
        2. read relevant chunk edges from cloud storage (include fake edges from big table)
        3. group nodes and edges based on level 2 ids (types.Agglomeration)
           optionally for each edge (v1,v2) active
           if parent(v1) == parent(v2) inactive otherwise
        returns dict of {id: types.Agglomeration}
        """
        # 1 level 2 ids
        if not layer_2:
            level2_ids = [types.empty_1d]
            for agglomeration_id in node_ids:
                layer_nodes_d = self._get_subgraph_higher_layer_nodes(
                    node_id=agglomeration_id,
                    bounding_box=chunk_utils.normalize_bounding_box(
                        self.meta, bbox, bbox_is_coordinate
                    ),
                    return_layers=[2],
                )
                level2_ids.append(layer_nodes_d[2])
            level2_ids = np.concatenate(level2_ids)
        else:
            level2_ids = node_ids

        # 2 edges from cloud storage
        chunk_ids = self.get_chunk_ids_from_node_ids(level2_ids)
        chunk_edge_dicts = mu.multithread_func(
            self.read_chunk_edges,
            np.array_split(np.unique(chunk_ids), 4),  # TODO
            n_threads=4,
            debug=False,
        )
        edges_dict = edge_utils.concatenate_chunk_edges(chunk_edge_dicts)
        edges = reduce(lambda x, y: x + y, edges_dict.values())
        # TODO include fake edges
        l2id_agglomeration_d = {}
        l2id_children_d = self.get_children(level2_ids)
        for l2id in l2id_children_d:
            supervoxels = l2id_children_d[l2id]
            in_, out_, cross_ = edge_utils.categorize_edges(
                self.meta, supervoxels, edges
            )
            l2id_agglomeration_d[l2id] = types.Agglomeration(
                l2id, supervoxels, in_, out_, cross_
            )
        return l2id_agglomeration_d

    def add_edges(
        self,
        user_id: str,
        atomic_edges: typing.Sequence[np.uint64],
        affinities: typing.Sequence[np.float32] = None,
        source_coord: typing.Sequence[int] = None,
        sink_coord: typing.Sequence[int] = None,
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
        source_ids: typing.Sequence[np.uint64] = None,
        sink_ids: typing.Sequence[np.uint64] = None,
        source_coords: typing.Sequence[typing.Sequence[int]] = None,
        sink_coords: typing.Sequence[typing.Sequence[int]] = None,
        atomic_edges: typing.Sequence[typing.Tuple[np.uint64, np.uint64]] = None,
        mincut: bool = True,
        bb_offset: typing.Tuple[int, int, int] = (240, 240, 24),
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

    # PRIVATE
    def _get_subgraph_higher_layer_nodes(
        self,
        node_id: basetypes.NODE_ID,
        bounding_box: typing.Optional[typing.Sequence[typing.Sequence[int]]],
        return_layers: typing.Sequence[int],
    ):
        def _get_subgraph_higher_layer_nodes_threaded(
            node_ids: typing.Iterable[np.uint64],
        ) -> typing.List[np.uint64]:
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

    def _run_multicut(
        self,
        source_ids: typing.Sequence[np.uint64],
        sink_ids: typing.Sequence[np.uint64],
        source_coords: typing.Sequence[typing.Sequence[int]],
        sink_coords: typing.Sequence[typing.Sequence[int]],
        bb_offset: typing.Tuple[int, int, int] = (120, 120, 12),
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

    def _get_bounding_l2_children(
        self,
        parent_ids: typing.Iterable,
        cache: typing.Dict[np.uint64, types.Node] = None,
    ) -> typing.Dict:
        """
        Helper function to get level 2 children IDs for each parent.
        `parent_ids` must contain node IDs at same layer.
        TODO describe algo
        """
        parents_layer = self.get_chunk_layer(parent_ids[0])
        parent_coords_d = {
            node_id: self.get_chunk_coordinates(node_id) for node_id in parent_ids
        }

        parent_bounding_chunk_ids = defaultdict(lambda: types.empty_1d)
        parent_layer_mask = {}

        parent_children_d = {
            parent_id: np.array([parent_id], dtype=basetypes.NODE_ID)
            for parent_id in parent_ids
        }

        children_layer = parents_layer - 1
        while children_layer >= 2:
            parent_masked_children_d = {}
            for parent_id, (X, Y, Z) in parent_coords_d.items():
                chunks = chunk_utils.get_bounding_children_chunks(
                    self.meta, parents_layer, (X, Y, Z), children_layer
                )
                parent_bounding_chunk_ids[parent_id] = np.array(
                    [
                        self.get_chunk_id(layer=children_layer, x=x, y=y, z=z)
                        for (x, y, z) in chunks
                    ]
                )
                children = parent_children_d[parent_id]
                layer_mask = self.get_chunk_layers(children) > children_layer
                parent_layer_mask[parent_id] = layer_mask
                parent_masked_children_d[parent_id] = children[layer_mask]

            children_ids = np.concatenate(list(parent_masked_children_d.values()))
            cache_node_ids = np.fromiter(cache.keys(), dtype=basetypes.NODE_ID)
            cache_mask = np.in1d(children_ids, cache_node_ids)
            child_grand_children_d = {
                child_id: cache[child_id].children
                for child_id in children_ids[cache_mask]
            }
            child_grand_children_d.update(self.get_children(children_ids[~cache_mask]))
            for parent_id, masked_children in parent_masked_children_d.items():
                bounding_chunk_ids = parent_bounding_chunk_ids[parent_id]
                grand_children = [types.empty_1d]
                for child in masked_children:
                    grand_children_ = child_grand_children_d[child]
                    chunk_ids = self.get_chunk_ids_from_node_ids(grand_children_)
                    grand_children_ = grand_children_[
                        np.in1d(chunk_ids, bounding_chunk_ids)
                    ]
                    grand_children.append(grand_children_)
                grand_children = np.concatenate(grand_children)

                unmasked_children = parent_children_d[parent_id]
                layer_mask = parent_layer_mask[parent_id]
                parent_children_d[parent_id] = np.concatenate(
                    [unmasked_children[~layer_mask], grand_children]
                )

            children_layer -= 1
        return parent_children_d

    # HELPERS / WRAPPERS
    def get_node_id(
        self,
        segment_id: np.uint64,
        chunk_id: typing.Optional[np.uint64] = None,
        layer: typing.Optional[int] = None,
        x: typing.Optional[int] = None,
        y: typing.Optional[int] = None,
        z: typing.Optional[int] = None,
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

    def get_chunk_layers(self, node_or_chunk_ids: typing.Sequence):
        return chunk_utils.get_chunk_layers(self.meta, node_or_chunk_ids)

    def get_chunk_coordinates(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_utils.get_chunk_coordinates(self.meta, node_or_chunk_id)

    def get_chunk_id(
        self,
        node_id: basetypes.NODE_ID = None,
        layer: typing.Optional[int] = None,
        x: typing.Optional[int] = 0,
        y: typing.Optional[int] = 0,
        z: typing.Optional[int] = 0,
    ):
        return chunk_utils.get_chunk_id(
            self.meta, node_id=node_id, layer=layer, x=x, y=y, z=z
        )

    def get_chunk_ids_from_node_ids(self, node_ids: typing.Sequence):
        return chunk_utils.get_chunk_ids_from_node_ids(self.meta, node_ids)

    def get_children_chunk_ids(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_children_chunk_ids(self.meta, node_or_chunk_id)

    def get_parent_chunk_id(
        self, node_or_chunk_id: basetypes.NODE_ID, parent_layer: int = None
    ):
        if not parent_layer:
            parent_layer = self.get_chunk_layer(node_or_chunk_id) + 1
        return chunk_hierarchy.get_parent_chunk_id(
            self.meta, node_or_chunk_id, parent_layer
        )

    def get_parent_chunk_ids(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_parent_chunk_ids(self.meta, node_or_chunk_id)

    def get_parent_chunk_id_dict(self, node_or_chunk_id: basetypes.NODE_ID):
        return chunk_hierarchy.get_parent_chunk_id_dict(self.meta, node_or_chunk_id)

    def get_cross_chunk_edges_layer(self, cross_edges: typing.Iterable):
        return edge_utils.get_cross_chunk_edges_layer(self.meta, cross_edges)

    def read_chunk_edges(self, chunk_ids: typing.Iterable, cv_threads: int = 1) -> dict:
        return get_chunk_edges(
            self.meta.data_source.EDGES,
            [self.get_chunk_coordinates(chunk_id) for chunk_id in chunk_ids],
            cv_threads=cv_threads,
        )


# TODO
# def read_consolidated_lock_timestamp
# def read_lock_timestamp
# def read_first_log_row
# def get_delta_roots
# def get_change_log
# def _setup_logger
# def read_logs
