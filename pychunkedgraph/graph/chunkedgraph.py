import time
import typing
import datetime
from functools import reduce

import numpy as np
from multiwrapper import multiprocessing_utils as mu

from . import types
from . import operation
from . import attributes
from . import exceptions
from .client import base
from .client.bigtable import BigTableClient
from .cache import CacheService
from .meta import ChunkedGraphMeta
from .meta import BackendClientInfo
from .utils import basetypes
from .utils import id_helpers
from .utils import generic as misc_utils
from .edges import Edges
from .edges import utils as edge_utils
from .chunks import utils as chunk_utils
from .chunks import hierarchy as chunk_hierarchy


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
            bt_client = BigTableClient(
                graph_id, config=client_info.CONFIG, graph_meta=meta
            )
            self._meta = meta
        else:
            bt_client = BigTableClient(graph_id, config=client_info.CONFIG)
            self._meta = bt_client.read_graph_meta()

        self._client = bt_client
        self._id_client = bt_client
        self._cache_service = None
        self.mock_edges = Edges([], [])  # for unit tests

    @property
    def meta(self) -> ChunkedGraphMeta:
        return self._meta

    @property
    def client(self) -> base.SimpleClient:
        return self._client

    @property
    def id_client(self) -> base.ClientWithIDGen:
        return self._id_client

    @property
    def cache(self):
        return self._cache_service

    @cache.setter
    def cache(self, cache_service: CacheService):
        self._cache_service = cache_service

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
        return id_helpers.get_atomic_id_from_coord(
            self.meta, self.get_root, x, y, z, parent_id, n_tries=n_tries,
        )

    def get_parents(
        self,
        node_ids: typing.Sequence[np.uint64],
        *,
        raw_only=False,
        current: bool = True,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ):
        """
        If current=True returns only the latest parents.
        Else all parents along with timestamps.
        """
        if raw_only or not self.cache:
            time_stamp = misc_utils.get_valid_timestamp(time_stamp)
            parent_rows = self.client.read_nodes(
                node_ids=node_ids,
                properties=attributes.Hierarchy.Parent,
                end_time=time_stamp,
                end_time_inclusive=True,
            )
            if not parent_rows:
                return types.empty_1d
            if current:
                return np.array(
                    [parent_rows[id_][0].value for id_ in node_ids],
                    dtype=basetypes.NODE_ID,
                )
            parents = []
            for id_ in node_ids:
                parents.append([(p.value, p.timestamp) for p in parent_rows[id_]])
            return parents
        return self.cache.parents_multiple(node_ids)

    def get_parent(
        self,
        node_id: np.uint64,
        *,
        raw_only=False,
        latest: bool = True,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Union[typing.List[typing.Tuple], np.uint64]:
        if raw_only or not self.cache:
            time_stamp = misc_utils.get_valid_timestamp(time_stamp)
            parents = self.client.read_node(
                node_id,
                properties=attributes.Hierarchy.Parent,
                end_time=time_stamp,
                end_time_inclusive=True,
            )
            if not parents:
                return None
            if latest:
                return parents[0].value
            return [(p.value, p.timestamp) for p in parents]
        return self.cache.parent(node_id)

    def get_children(
        self,
        node_id_or_ids: typing.Union[typing.Iterable[np.uint64], np.uint64],
        *,
        raw_only=False,
        flatten: bool = False,
    ) -> typing.Union[typing.Dict, np.ndarray]:
        """
        Children for the specified NodeID or NodeIDs.
        If flatten == True, an array is returned, else a dict {node_id: children}.
        """
        if np.isscalar(node_id_or_ids):
            if raw_only or not self.cache:
                children = self.client.read_node(
                    node_id=node_id_or_ids, properties=attributes.Hierarchy.Child
                )
                if not children:
                    return types.empty_1d.copy()
                return children[0].value
            return self.cache.children(node_id_or_ids)
        node_children_d = self._get_children_multiple(node_id_or_ids, raw_only=raw_only)
        if flatten:
            if not node_children_d:
                return types.empty_1d.copy()
            return np.concatenate([*node_children_d.values()])
        return node_children_d

    def _get_children_multiple(
        self, node_ids: typing.Iterable[np.uint64], *, raw_only=False,
    ) -> typing.Dict:
        if raw_only or not self.cache:
            node_children_d = self.client.read_nodes(
                node_ids=node_ids, properties=attributes.Hierarchy.Child
            )
            return {
                x: node_children_d[x][0].value
                if x in node_children_d
                else types.empty_1d.copy()
                for x in node_ids
            }
        return self.cache.children_multiple(node_ids)

    def get_atomic_cross_edges(
        self, l2_ids: typing.Iterable, *, raw_only=False,
    ) -> typing.Dict[np.uint64, typing.Dict[int, typing.Iterable]]:
        """Returns cross edges for level 2 IDs."""
        if raw_only or not self.cache:
            node_edges_d_d = self.client.read_nodes(
                node_ids=l2_ids,
                properties=[
                    attributes.Connectivity.CrossChunkEdge[l]
                    for l in range(2, self.meta.layer_count)
                ],
            )
            result = {}
            for id_ in l2_ids:
                try:
                    result[id_] = {
                        prop.index: val[0].value.copy()
                        for prop, val in node_edges_d_d[id_].items()
                    }
                except KeyError:
                    result[id_] = {}
            return result
        return self.cache.atomic_cross_edges_multiple(l2_ids)

    def get_cross_chunk_edges(
        self, node_ids: np.ndarray, uplift=True, all_layers=False
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
        """
        result = {}
        if not node_ids.size:
            return result

        from .utils.context_managers import TimeIt

        with TimeIt(f"_get_bounding_l2_children {len(node_ids)}"):
            node_l2ids_d = self._get_bounding_l2_children(node_ids)
        l2_edges_d_d = self.get_atomic_cross_edges(
            np.concatenate(list(node_l2ids_d.values()))
        )
        for node_id in node_ids:
            l2_edges_ds = [l2_edges_d_d[l2_id] for l2_id in node_l2ids_d[node_id]]
            if all_layers:
                result[node_id] = edge_utils.concatenate_cross_edge_dicts(l2_edges_ds)
            else:
                result[node_id] = self._get_min_layer_cross_edges(
                    node_id, l2_edges_ds, uplift=uplift
                )
        return result

    def _get_min_layer_cross_edges(
        self,
        node_id: basetypes.NODE_ID,
        l2id_atomic_cross_edges_ds: typing.Iterable,
        uplift=True,
    ) -> typing.Dict[int, typing.Iterable]:
        """
        Find edges at relevant min_layer >= node_layer.
        `l2id_atomic_cross_edges_ds` is a list of atomic cross edges of
        level 2 IDs that are descendants of `node_id`.
        """
        min_layer, edges = edge_utils.filter_min_layer_cross_edges_multiple(
            self.meta, l2id_atomic_cross_edges_ds, self.get_chunk_layer(node_id)
        )
        if self.get_chunk_layer(node_id) < min_layer:
            # cross edges irrelevant
            return {min_layer: types.empty_2d}
        if not uplift:
            return {min_layer: edges}
        node_root_id = node_id
        node_root_id = self.get_root(node_id, stop_layer=min_layer, ceil=False)
        edges[:, 0] = node_root_id
        edges[:, 1] = self.get_roots(edges[:, 1], stop_layer=min_layer, ceil=False)
        return {min_layer: np.unique(edges, axis=0) if edges.size else types.empty_2d}

    def get_roots(
        self,
        node_ids: typing.Sequence[np.uint64],
        *,
        time_stamp: typing.Optional[datetime.datetime] = None,
        stop_layer: int = None,
        ceil: bool = True,
        n_tries: int = 1,
    ) -> typing.Union[np.ndarray, typing.Dict[int, np.ndarray]]:
        """
        Returns node IDs at the root_layer/ <= stop_layer.
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        assert stop_layer <= self.meta.layer_count
        layer_mask = np.ones(len(node_ids), dtype=np.bool)

        for _ in range(n_tries):
            layer_mask[self.get_chunk_layers(node_ids) >= stop_layer] = False
            parent_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
            for _ in range(int(stop_layer + 1)):
                filtered_ids = parent_ids[layer_mask]
                unique_ids, inverse = np.unique(filtered_ids, return_inverse=True)
                temp_ids = self.get_parents(unique_ids, time_stamp=time_stamp)
                if not temp_ids.size:
                    break
                else:
                    temp_ids_i = temp_ids[inverse]
                    new_layer_mask = layer_mask.copy()
                    new_layer_mask[new_layer_mask] = (
                        self.get_chunk_layers(temp_ids_i) < stop_layer
                    )
                    if not ceil:
                        rev_m = self.get_chunk_layers(temp_ids_i) > stop_layer
                        temp_ids_i[rev_m] = filtered_ids[rev_m]

                    parent_ids[layer_mask] = temp_ids_i
                    layer_mask = new_layer_mask

                    if np.all(~layer_mask):
                        return parent_ids

            if not ceil and np.all(self.get_chunk_layers(parent_ids) >= stop_layer):
                return parent_ids
            elif ceil:
                return parent_ids
            else:
                time.sleep(0.5)
        return parent_ids

    def get_root(
        self,
        node_id: np.uint64,
        *,
        time_stamp: typing.Optional[datetime.datetime] = None,
        get_all_parents: bool = False,
        stop_layer: int = None,
        ceil: bool = True,
        n_tries: int = 1,
    ) -> typing.Union[typing.List[np.uint64], np.uint64]:
        """Takes a node id and returns the associated agglomeration ids."""
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        parent_id = node_id
        all_parent_ids = []
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        if self.get_chunk_layer(parent_id) == stop_layer:
            return (
                np.array([node_id], dtype=basetypes.NODE_ID)
                if get_all_parents
                else node_id
            )

        for _ in range(n_tries):
            parent_id = node_id
            for _ in range(self.get_chunk_layer(node_id), int(stop_layer + 1)):
                temp_parent_id = self.get_parent(parent_id, time_stamp=time_stamp)
                if temp_parent_id is None:
                    break
                else:
                    parent_id = temp_parent_id

                    if self.get_chunk_layer(parent_id) >= stop_layer:
                        if self.get_chunk_layer(parent_id) == stop_layer:
                            all_parent_ids.append(parent_id)
                        elif ceil:
                            all_parent_ids.append(parent_id)
                        break
                    else:
                        all_parent_ids.append(parent_id)

            if self.get_chunk_layer(parent_id) >= stop_layer:
                break
            else:
                time.sleep(0.5)

        if self.get_chunk_layer(parent_id) < stop_layer:
            raise exceptions.ChunkedGraphError(
                f"Cannot find root id {node_id}, {stop_layer}, {time_stamp}"
            )

        if get_all_parents:
            return np.array(all_parent_ids, dtype=basetypes.NODE_ID)
        else:
            if len(all_parent_ids) == 0:
                return node_id
            else:
                return all_parent_ids[-1]

    def get_all_parents_dict(
        self,
        node_id: basetypes.NODE_ID,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Dict:
        """Takes a node id and returns all parents up to root."""
        parent_ids = self.get_root(
            node_id=node_id, time_stamp=time_stamp, get_all_parents=True
        )
        return dict(zip(self.get_chunk_layers(parent_ids), parent_ids))

    def get_subgraph(
        self,
        node_id_or_ids: typing.Union[np.uint64, typing.Iterable],
        bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
        return_layers: typing.List = [2],
        nodes_only: bool = False,
        edges_only: bool = False,
        leaves_only: bool = False,
    ) -> typing.Tuple[typing.Dict, typing.Dict, Edges]:
        """TODO docs"""
        single = False
        node_ids = node_id_or_ids
        bbox = chunk_utils.normalize_bounding_box(self.meta, bbox, bbox_is_coordinate)
        if isinstance(node_id_or_ids, np.uint64) or isinstance(node_id_or_ids, int):
            single = True
            node_ids = [node_id_or_ids]
        layer_nodes_d = {}
        for node_id in node_ids:
            layer_nodes_d[node_id] = self._get_subgraph_higher_layer_nodes(
                node_id=node_id, bounding_box=bbox, return_layers=return_layers,
            )
        if nodes_only:
            if single:
                return layer_nodes_d[node_id_or_ids]
            return layer_nodes_d
        level2_ids = [types.empty_1d]
        for node_id in node_ids:
            layer_nodes_d = self._get_subgraph_higher_layer_nodes(
                node_id=node_id, bounding_box=bbox, return_layers=[2],
            )
            level2_ids.append(layer_nodes_d[2])
        level2_ids = np.concatenate(level2_ids)
        if leaves_only:
            return self.get_children(level2_ids, flatten=True)
        if edges_only:
            return self.get_l2_agglomerations(level2_ids, edges_only=True)
        l2id_agglomeration_d, edges = self.get_l2_agglomerations(level2_ids)
        return l2id_agglomeration_d, edges

    def get_fake_edges(
        self, chunk_ids: np.ndarray, time_stamp: datetime.datetime = None
    ) -> typing.Dict:
        result = {}
        fake_edges_d = self.client.read_nodes(
            node_ids=chunk_ids,
            properties=attributes.Connectivity.FakeEdges,
            end_time=time_stamp,
            end_time_inclusive=True,
            fake_edges=True,
        )
        for id_, val in fake_edges_d.items():
            edges = np.concatenate(
                [np.array(e.value, dtype=basetypes.NODE_ID) for e in val]
            )
            result[id_] = Edges(edges[:, 0], edges[:, 1], fake_edges=True)
        return result

    def get_l2_agglomerations(
        self, level2_ids: np.ndarray, edges_only: bool = False
    ) -> typing.Tuple[typing.Dict[int, types.Agglomeration], np.ndarray]:
        """
        Children of Level 2 Node IDs and edges.
        Edges are read from cloud storage.
        """
        chunk_ids = np.unique(self.get_chunk_ids_from_node_ids(level2_ids))
        chunk_edge_dicts = mu.multithread_func(
            self.read_chunk_edges,
            np.array_split(chunk_ids, 8),  # TODO hardcoded
            n_threads=8,
            debug=False,
        )
        edges_d = edge_utils.concatenate_chunk_edges(chunk_edge_dicts)
        all_chunk_edges = reduce(lambda x, y: x + y, edges_d.values(), Edges([], []))
        print("all_chunk_edges b", len(all_chunk_edges))
        all_chunk_edges += reduce(
            lambda x, y: x + y, self.get_fake_edges(chunk_ids).values(), Edges([], [])
        )
        print("all_chunk_edges a", len(all_chunk_edges))
        if edges_only:
            if len(self.mock_edges):
                all_chunk_edges = self.mock_edges.get_pairs()
            else:
                all_chunk_edges = all_chunk_edges.get_pairs()
            supervoxels = self.get_children(level2_ids, flatten=True)
            mask0 = np.in1d(all_chunk_edges[:, 0], supervoxels)
            mask1 = np.in1d(all_chunk_edges[:, 1], supervoxels)
            mask2 = np.in1d(all_chunk_edges[:, 1], supervoxels)
            mask3 = np.in1d(all_chunk_edges[:, 0], supervoxels)
            return all_chunk_edges[mask0 & mask1 | mask2 & mask3]
        in_edges = set()
        out_edges = set()
        cross_edges = set()
        # TODO include fake edges
        l2id_agglomeration_d = {}
        l2id_children_d = self.get_children(level2_ids)
        for l2id in l2id_children_d:
            supervoxels = l2id_children_d[l2id]
            in_, out_, cross_ = edge_utils.categorize_edges(
                self.meta, supervoxels, all_chunk_edges
            )
            l2id_agglomeration_d[l2id] = types.Agglomeration(
                l2id, supervoxels, in_, out_, cross_
            )
            in_edges.add(in_)
            out_edges.add(out_)
            cross_edges.add(cross_)
        in_edges = reduce(lambda x, y: x + y, in_edges)
        out_edges = reduce(lambda x, y: x + y, out_edges)
        cross_edges = reduce(lambda x, y: x + y, cross_edges)
        return (
            l2id_agglomeration_d,
            (self.mock_edges,)
            if len(self.mock_edges)
            else (in_edges, out_edges, cross_edges),
        )

    def get_node_timestamps(
        self, node_ids: typing.Sequence[np.uint64]
    ) -> typing.Iterable:
        """
        The timestamp of the children column can be assumed
        to be the timestamp at which the node ID was created.
        """
        children = self.client.read_nodes(
            node_ids=node_ids, properties=attributes.Hierarchy.Child
        )

        if not children:
            return np.array([], dtype=np.datetime64)
        return np.array(
            [children[x][0].timestamp for x in node_ids], dtype=np.datetime64
        )

    # OPERATIONS
    def add_edges(
        self,
        user_id: str,
        atomic_edges: typing.Sequence[np.uint64],
        *,
        affinities: typing.Sequence[np.float32] = None,
        source_coords: typing.Sequence[int] = None,
        sink_coords: typing.Sequence[int] = None,
    ) -> operation.GraphEditOperation.Result:
        """
        Adds an edge to the chunkedgraph
        Multi-user safe through locking of the root node
        This function acquires a lock and ensures that it still owns the
        lock before executing the write.
        :return: GraphEditOperation.Result
        """
        return operation.MergeOperation(
            self,
            user_id=user_id,
            added_edges=atomic_edges,
            affinities=affinities,
            source_coords=source_coords,
            sink_coords=sink_coords,
        ).execute()

    def remove_edges(
        self,
        user_id: str,
        *,
        atomic_edges: typing.Sequence[typing.Tuple[np.uint64, np.uint64]] = None,
        source_ids: typing.Sequence[np.uint64] = None,
        sink_ids: typing.Sequence[np.uint64] = None,
        source_coords: typing.Sequence[typing.Sequence[int]] = None,
        sink_coords: typing.Sequence[typing.Sequence[int]] = None,
        mincut: bool = True,
        bb_offset: typing.Tuple[int, int, int] = (240, 240, 24),
    ) -> operation.GraphEditOperation.Result:
        """
        Removes edges - either directly or after applying a mincut
        Multi-user safe through locking of the root node
        This function acquires a lock and ensures that it still owns the
        lock before executing the write.
        :param atomic_edges: list of 2 uint64
        :param bb_offset: list of 3 ints
            [x, y, z] bounding box padding beyond box spanned by coordinates
        :return: GraphEditOperation.Result
        """
        source_ids = [source_ids] if np.isscalar(source_ids) else source_ids
        sink_ids = [sink_ids] if np.isscalar(sink_ids) else sink_ids
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
            if len(source_ids) != len(sink_ids):
                raise exceptions.PreconditionError(
                    "Split operation require the same number of source and sink IDs"
                )
            atomic_edges = np.array(
                [source_ids, sink_ids], dtype=basetypes.NODE_ID
            ).transpose()
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
        node_id: np.uint64,
        bounding_box: typing.Optional[typing.Sequence[typing.Sequence[int]]],
        return_layers: typing.Sequence[int],
    ):
        def _get_subgraph_higher_layer_nodes_threaded(
            node_ids: typing.Iterable[np.uint64],
        ) -> typing.List[np.uint64]:
            children = self.get_children(node_ids, flatten=True)
            if len(children) > 0 and bounding_box is not None:
                chunk_coordinates = np.array(
                    [self.get_chunk_coordinates(c) for c in children]
                )
                child_layers = self.get_chunk_layers(children)
                adapt_child_layers = child_layers - 2
                adapt_child_layers[adapt_child_layers < 0] = 0

                fanout = self.meta.graph_config.FANOUT
                bounding_box_layer = (
                    bounding_box[None] / (fanout ** adapt_child_layers)[:, None, None]
                )

                bound_check = np.array(
                    [
                        np.all(chunk_coordinates < bounding_box_layer[:, 1], axis=1),
                        np.all(
                            chunk_coordinates + 1 > bounding_box_layer[:, 0], axis=1
                        ),
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

            from itertools import chain

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

    def _get_bounding_l2_children(self, parent_ids: typing.Iterable) -> typing.Dict:
        """
        Helper function to get level 2 children IDs for each parent.
        `parent_ids` must contain node IDs at same layer.
        TODO what have i done (describe algo)
        """
        parents_layer = self.get_chunk_layer(parent_ids[0])
        parent_coords_d = {
            node_id: self.get_chunk_coordinates(node_id) for node_id in parent_ids
        }

        from collections import defaultdict

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
                    ],
                    dtype=basetypes.CHUNK_ID,
                )
                children = parent_children_d[parent_id]
                layer_mask = self.get_chunk_layers(children) > children_layer
                parent_layer_mask[parent_id] = layer_mask
                parent_masked_children_d[parent_id] = children[layer_mask]

            children_ids = np.concatenate(list(parent_masked_children_d.values()))
            child_grand_children_d = self.get_children(children_ids)
            for parent_id, masked_children in parent_masked_children_d.items():
                bounding_chunk_ids = parent_bounding_chunk_ids[parent_id]
                grand_children = [types.empty_1d]
                for child in masked_children:
                    grand_children_ = child_grand_children_d[child]
                    mask = self.get_chunk_layers(grand_children_) == children_layer
                    masked_grand_children_ = grand_children_[mask]
                    chunk_ids = self.get_chunk_ids_from_node_ids(masked_grand_children_)
                    masked_grand_children_ = masked_grand_children_[
                        np.in1d(chunk_ids, bounding_chunk_ids)
                    ]
                    grand_children_ = np.concatenate(
                        [masked_grand_children_, grand_children_[~mask]]
                    )
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

    def get_serialized_info(self):
        return {
            "graph_id": self.meta.graph_config.ID_PREFIX + self.meta.graph_config.ID
        }

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

    def read_chunk_edges(
        self, chunk_ids: typing.Iterable, cv_threads: int = 1
    ) -> typing.Dict:
        from ..io.edges import get_chunk_edges

        return get_chunk_edges(
            self.meta.data_source.EDGES,
            [self.get_chunk_coordinates(chunk_id) for chunk_id in chunk_ids],
            cv_threads=cv_threads,
        )
