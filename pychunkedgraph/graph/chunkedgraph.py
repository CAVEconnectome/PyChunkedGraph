import time
import typing
import datetime

import numpy as np

from . import types
from . import operation
from . import attributes
from . import exceptions
from .client import base
from .client import BigTableClient
from .client import BackendClientInfo
from .client import get_default_client_info
from .cache import CacheService
from .meta import ChunkedGraphMeta
from .meta import VirtualChunkedGraphMeta
from .utils import basetypes
from .utils import id_helpers
from .utils import generic as misc_utils
from .utils.context_managers import TimeIt
from .edges import Edges
from .edges import utils as edge_utils
from .chunks import utils as chunk_utils
from .chunks import hierarchy as chunk_hierarchy


class ChunkedGraph:
    def __init__(
        self,
        *,
        graph_id: str = None,
        meta: ChunkedGraphMeta = None,
        client_info: BackendClientInfo = get_default_client_info(),
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
                graph_id,
                config=client_info.CONFIG,
                graph_meta=meta,
            )
            self._meta = meta
        else:
            bt_client = BigTableClient(graph_id, config=client_info.CONFIG)
            self._meta = bt_client.read_graph_meta()

        self._client = bt_client
        self._id_client = bt_client
        self._cache_service = None
        self.mock_edges = None  # hack for unit tests

    @property
    def meta(self) -> ChunkedGraphMeta:
        if self._meta is None:
            self._meta = self.client.read_graph_meta()
        return self._meta

    @property
    def graph_id(self) -> str:
        return self.meta.graph_config.ID_PREFIX + self.meta.graph_config.ID

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
        self.client.create_graph(self.meta)

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
        self,
        x: int,
        y: int,
        z: int,
        parent_id: np.uint64,
        n_tries: int = 5,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> np.uint64:
        """Determines atomic id given a coordinate."""
        if self.get_chunk_layer(parent_id) == 1:
            return parent_id
        return id_helpers.get_atomic_id_from_coord(
            self.meta,
            self.get_root,
            x,
            y,
            z,
            parent_id,
            n_tries=n_tries,
            time_stamp=time_stamp,
        )

    def get_parents(
        self,
        node_ids: typing.Sequence[np.uint64],
        *,
        raw_only=False,
        current: bool = True,
        fail_to_zero: bool = False,
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

            parents = []
            if current:
                for id_ in node_ids:
                    try:
                        parents.append(parent_rows[id_][0].value)
                    except KeyError:
                        if fail_to_zero:
                            parents.append(0)
                        else:
                            raise KeyError
                parents = np.array(parents, dtype=basetypes.NODE_ID)
            else:
                for id_ in node_ids:
                    try:
                        parents.append(
                            [(p.value, p.timestamp) for p in parent_rows[id_]]
                        )
                    except KeyError:
                        if fail_to_zero:
                            parents.append([(0, datetime.datetime.fromtimestamp(0))])
                        else:
                            raise KeyError
            return parents
        return self.cache.parents_multiple(node_ids, time_stamp=time_stamp)

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
        return self.cache.parent(node_id, time_stamp=time_stamp)

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
        self, node_ids: typing.Iterable[np.uint64], *, raw_only=False
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
        self, l2_ids: typing.Iterable, *, raw_only=False
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

        node_l2ids_d = {}
        layers_ = self.get_chunk_layers(node_ids)
        for l in set(layers_):
            node_l2ids_d.update(self._bounding_l2_children(node_ids[layers_ == l]))
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
            return {self.get_chunk_layer(node_id): types.empty_2d}
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
        assert_roots: bool = False,
        time_stamp: typing.Optional[datetime.datetime] = None,
        stop_layer: int = None,
        ceil: bool = True,
        n_tries: int = 1,
    ) -> typing.Union[np.ndarray, typing.Dict[int, np.ndarray]]:
        """
        Returns node IDs at the root_layer/ <= stop_layer.
        Use `assert_roots=True` to ensure returned IDs are at root level.
        When `assert_roots=False`, returns highest available IDs and
        cases where there are no root IDs are silently ignored.
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        stop_layer = self.meta.layer_count if not stop_layer else stop_layer
        assert stop_layer <= self.meta.layer_count
        layer_mask = np.ones(len(node_ids), dtype=bool)

        for _ in range(n_tries):
            chunk_layers = self.get_chunk_layers(node_ids)
            layer_mask[chunk_layers >= stop_layer] = False
            layer_mask[node_ids == 0] = False

            parent_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
            for _ in range(int(stop_layer + 1)):
                filtered_ids = parent_ids[layer_mask]
                unique_ids, inverse = np.unique(filtered_ids, return_inverse=True)
                temp_ids = self.get_parents(
                    unique_ids, time_stamp=time_stamp, fail_to_zero=True
                )
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
                        if assert_roots:
                            assert not np.any(
                                self.get_chunk_layers(parent_ids)
                                < self.meta.layer_count
                            ), "roots not found for some IDs"
                        return parent_ids

            if not ceil and np.all(
                self.get_chunk_layers(parent_ids[parent_ids != 0]) >= stop_layer
            ):
                if assert_roots:
                    assert not np.any(
                        self.get_chunk_layers(parent_ids) < self.meta.layer_count
                    ), "roots not found for some IDs"
                return parent_ids
            elif ceil:
                if assert_roots:
                    assert not np.any(
                        self.get_chunk_layers(parent_ids) < self.meta.layer_count
                    ), "roots not found for some IDs"
                return parent_ids
            else:
                time.sleep(0.5)
        if assert_roots:
            assert not np.any(
                self.get_chunk_layers(parent_ids) < self.meta.layer_count
            ), "roots not found for some IDs"
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

    def is_latest_roots(
        self,
        root_ids: typing.Iterable,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Iterable:
        """Determines whether root ids are superseeded."""
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)

        row_dict = self.client.read_nodes(
            node_ids=root_ids,
            properties=attributes.Hierarchy.NewParent,
            end_time=time_stamp,
        )
        return ~np.isin(root_ids, list(row_dict.keys()))

    def get_all_parents_dict(
        self,
        node_id: basetypes.NODE_ID,
        *,
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
        return_flattened: bool = False,
    ) -> typing.Tuple[typing.Dict, typing.Dict, Edges]:
        """
        Generic subgraph method.
        """
        from .subgraph import get_subgraph_nodes
        from .subgraph import get_subgraph_edges_and_leaves

        if nodes_only:
            return get_subgraph_nodes(
                self,
                node_id_or_ids,
                bbox,
                bbox_is_coordinate,
                return_layers,
                return_flattened=return_flattened,
            )
        return get_subgraph_edges_and_leaves(
            self, node_id_or_ids, bbox, bbox_is_coordinate, edges_only, leaves_only
        )

    def get_subgraph_nodes(
        self,
        node_id_or_ids: typing.Union[np.uint64, typing.Iterable],
        bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
        return_layers: typing.List = [2],
        serializable: bool = False,
        return_flattened: bool = False,
    ) -> typing.Tuple[typing.Dict, typing.Dict, Edges]:
        """
        Get the children of `node_ids` that are at each of
        return_layers within the specified bounding box.
        """
        from .subgraph import get_subgraph_nodes

        return get_subgraph_nodes(
            self,
            node_id_or_ids,
            bbox,
            bbox_is_coordinate,
            return_layers,
            serializable=serializable,
            return_flattened=return_flattened,
        )

    def get_subgraph_edges(
        self,
        node_id_or_ids: typing.Union[np.uint64, typing.Iterable],
        bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
    ):
        """
        Get the atomic edges of the `node_ids` within the specified bounding box.
        """
        from .subgraph import get_subgraph_edges_and_leaves

        return get_subgraph_edges_and_leaves(
            self, node_id_or_ids, bbox, bbox_is_coordinate, True, False
        )

    def get_subgraph_leaves(
        self,
        node_id_or_ids: typing.Union[np.uint64, typing.Iterable],
        bbox: typing.Optional[typing.Sequence[typing.Sequence[int]]] = None,
        bbox_is_coordinate: bool = False,
    ):
        """
        Get the supervoxels of the `node_ids` within the specified bounding box.
        """
        from .subgraph import get_subgraph_edges_and_leaves

        return get_subgraph_edges_and_leaves(
            self, node_id_or_ids, bbox, bbox_is_coordinate, False, True
        )

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
        from itertools import chain
        from functools import reduce
        from .misc import get_agglomerations

        chunk_ids = np.unique(self.get_chunk_ids_from_node_ids(level2_ids))
        # google does not provide a storage emulator at the moment
        # this is an ugly hack to avoid permission issues in tests
        # TODO find a better way to test
        edges_d = {}
        if self.mock_edges is None:
            with TimeIt(f"reading {len(chunk_ids)} chunks"):
                edges_d = self.read_chunk_edges(chunk_ids)

        fake_edges = self.get_fake_edges(chunk_ids)
        all_chunk_edges = reduce(
            lambda x, y: x + y,
            chain(edges_d.values(), fake_edges.values()),
            Edges([], []),
        )

        if edges_only:
            if self.mock_edges is not None:
                all_chunk_edges = self.mock_edges.get_pairs()
            else:
                all_chunk_edges = all_chunk_edges.get_pairs()
            supervoxels = self.get_children(level2_ids, flatten=True)
            mask0 = np.in1d(all_chunk_edges[:, 0], supervoxels)
            mask1 = np.in1d(all_chunk_edges[:, 1], supervoxels)
            return all_chunk_edges[mask0 & mask1]

        with TimeIt(f"categorize_edges"):
            l2id_children_d = self.get_children(level2_ids)
            sv_parent_d = {}
            supervoxels = []
            for l2id in l2id_children_d:
                svs = l2id_children_d[l2id]
                sv_parent_d.update(dict(zip(svs.tolist(), [l2id] * len(svs))))
                supervoxels.append(svs)

            supervoxels = np.concatenate(supervoxels)

            def f(x):
                return sv_parent_d.get(x, x)

            get_sv_parents = np.vectorize(f, otypes=[np.uint64])
            in_edges, out_edges, cross_edges = edge_utils.categorize_edges_v2(
                self.meta,
                supervoxels,
                all_chunk_edges,
                l2id_children_d,
                get_sv_parents,
            )

        agglomeration_d = get_agglomerations(
            l2id_children_d, in_edges, out_edges, cross_edges, get_sv_parents
        )
        return (
            agglomeration_d,
            (self.mock_edges,)
            if self.mock_edges is not None
            else (in_edges, out_edges, cross_edges),
        )

    def get_node_timestamps(
        self, node_ids: typing.Sequence[np.uint64], return_numpy=True
    ) -> typing.Iterable:
        """
        The timestamp of the children column can be assumed
        to be the timestamp at which the node ID was created.
        """
        children = self.client.read_nodes(
            node_ids=node_ids, properties=attributes.Hierarchy.Child
        )

        if not children:
            if return_numpy:
                return np.array([], dtype=np.datetime64)
            return []
        if return_numpy:
            return np.array(
                [children[x][0].timestamp for x in node_ids], dtype=np.datetime64
            )
        return [children[x][0].timestamp for x in node_ids]

    # OPERATIONS
    def add_edges(
        self,
        user_id: str,
        atomic_edges: typing.Sequence[np.uint64],
        *,
        affinities: typing.Sequence[np.float32] = None,
        source_coords: typing.Sequence[int] = None,
        sink_coords: typing.Sequence[int] = None,
        allow_same_segment_merge: typing.Optional[bool] = False,
    ) -> operation.GraphEditOperation.Result:
        """
        Adds an edge to the chunkedgraph
        Multi-user safe through locking of the root node
        This function acquires a lock and ensures that it still owns the
        lock before executing the write.
        :return: GraphEditOperation.Result
        """
        with TimeIt("MergeOperation.execute()"):
            return operation.MergeOperation(
                self,
                user_id=user_id,
                added_edges=atomic_edges,
                affinities=affinities,
                source_coords=source_coords,
                sink_coords=sink_coords,
                allow_same_segment_merge=allow_same_segment_merge,
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
        path_augment: bool = True,
        disallow_isolating_cut: bool = True,
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
            with TimeIt("MulticutOperation.execute()"):
                return operation.MulticutOperation(
                    self,
                    user_id=user_id,
                    source_ids=source_ids,
                    sink_ids=sink_ids,
                    source_coords=source_coords,
                    sink_coords=sink_coords,
                    bbox_offset=bb_offset,
                    path_augment=path_augment,
                    disallow_isolating_cut=disallow_isolating_cut,
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
        with TimeIt("SplitOperation.execute()"):
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
        """Applies the inverse of a previous GraphEditOperation
        :param user_id: str
        :param operation_id: operation_id to be inverted
        :return: GraphEditOperation.Result
        """
        return operation.GraphEditOperation.undo_operation(
            self,
            user_id=user_id,
            operation_id=operation_id,
            multicut_as_split=True,
        ).execute()

    def redo_operation(
        self, user_id: str, operation_id: np.uint64
    ) -> operation.GraphEditOperation.Result:
        """Re-applies a previous GraphEditOperation
        :param user_id: str
        :param operation_id: operation_id to be repeated
        :return: GraphEditOperation.Result
        """
        return operation.GraphEditOperation.redo_operation(
            self,
            user_id=user_id,
            operation_id=operation_id,
            multicut_as_split=True,
        ).execute()

    # PRIVATE
    def _bounding_l2_children(self, parent_ids: typing.Iterable) -> typing.Dict:
        """
        Helper function to get level 2 children IDs for each parent.
        `parent_ids` must be node IDs at same layer.
        TODO what have i done (describe algo)
        """
        from collections import defaultdict

        layers = self.get_chunk_layers(parent_ids)
        assert np.all(layers == layers[0])

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
                coords = chunk_utils.get_bounding_children_chunks(
                    self.meta,
                    parents_layer,
                    (X, Y, Z),
                    children_layer,
                    return_unique=False,
                )
                chunks_ids = chunk_utils.get_chunk_ids_from_coords(
                    self.meta, children_layer, coords
                )
                parent_bounding_chunk_ids[parent_id] = chunks_ids
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

    def is_root(self, node_id: basetypes.NODE_ID) -> bool:
        return self.get_chunk_layer(node_id) == self.meta.layer_count

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

    def read_chunk_edges(self, chunk_ids: typing.Iterable) -> typing.Dict:
        from ..io.edges import get_chunk_edges

        return get_chunk_edges(
            self.meta.data_source.EDGES,
            [self.get_chunk_coordinates(chunk_id) for chunk_id in chunk_ids],
        )

    def get_proofread_root_ids(
        self,
        start_time: typing.Optional[datetime.datetime] = None,
        end_time: typing.Optional[datetime.datetime] = None,
    ):
        from .misc import get_proofread_root_ids

        return get_proofread_root_ids(self, start_time, end_time)

    def get_earliest_timestamp(self):
        from datetime import timedelta

        for op_id in range(100):
            _, timestamp = self.client.read_log_entry(op_id)
            if timestamp is not None:
                return timestamp - timedelta(milliseconds=500)


class VirtualChunkedGraph(ChunkedGraph):
    """
    Virtual chunkedgraph points to a chunkedgraph and is read-only.
    """

    def __init__(
        self,
        virtual_graph_id: str,
        *,
        target_graph_id: typing.Optional[str] = None,
        timestamp_virtual: typing.Optional[datetime.datetime] = None,
        client_info: BackendClientInfo = get_default_client_info(),
    ):
        self._meta = None
        self._client = BigTableClient(virtual_graph_id, config=client_info.CONFIG)
        self._cache_service = None

        if timestamp_virtual is not None:
            timestamp_virtual = misc_utils.get_valid_timestamp(timestamp_virtual)
        if target_graph_id is not None:
            self._meta = VirtualChunkedGraphMeta(target_graph_id, timestamp_virtual)
        else:
            target_graph_id = self.meta.target_graph_id
            timestamp_virtual = self.meta.timestamp
        self._target_graph_client = BigTableClient(
            target_graph_id, config=client_info.CONFIG, timestamp=timestamp_virtual
        )

    @property
    def client(self) -> base.SimpleClient:
        return self._target_graph_client

    @property
    def meta(self) -> VirtualChunkedGraphMeta:
        if self._meta is None:
            self._meta = self._client.read_graph_meta()
        return self._meta

    def create(self):
        self._client.create_graph(self.meta)

    def update_meta(self, meta: VirtualChunkedGraphMeta):
        self._client.update_graph_meta(meta)
