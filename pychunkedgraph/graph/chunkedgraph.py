# pylint: disable=invalid-name, missing-docstring, too-many-lines, import-outside-toplevel, unsupported-binary-operation

import time
import typing
import datetime
from itertools import chain
from functools import reduce

import numpy as np
from pychunkedgraph import __version__

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
from .utils import basetypes
from .utils import id_helpers
from .utils import serializers
from .utils import generic as misc_utils
from .edges import Edges
from .edges import utils as edge_utils
from .chunks import utils as chunk_utils
from .chunks import hierarchy as chunk_hierarchy
from .subgraph import get_subgraph_nodes
from .subgraph import get_subgraph_edges_and_leaves


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
        # create client based on type
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
        self.mock_edges = None  # hack for unit tests

    @property
    def meta(self) -> ChunkedGraphMeta:
        return self._meta

    @property
    def graph_id(self) -> str:
        return self.meta.graph_config.ID_PREFIX + self.meta.graph_config.ID

    @property
    def version(self) -> str:
        return self.client.read_graph_version()

    @property
    def client(self) -> BigTableClient:
        return self._client

    @property
    def id_client(self) -> base.ClientWithIDGen:
        return self._id_client

    @property
    def cache(self):
        return self._cache_service

    @property
    def segmentation_resolution(self) -> np.ndarray:
        return np.array(self.meta.ws_cv.scale["resolution"])

    @cache.setter
    def cache(self, cache_service: CacheService):
        self._cache_service = cache_service

    def create(self):
        """Creates the graph in storage client and stores meta."""
        self._client.create_graph(self._meta, version=__version__)

    def update_meta(self, meta: ChunkedGraphMeta, overwrite: bool):
        """Update meta of an already existing graph."""
        self.client.update_graph_meta(meta, overwrite=overwrite)

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
        root_chunk = layer == self.meta.layer_count
        max_id = self.id_client.get_max_node_id(
            chunk_id=chunk_id, root_chunk=root_chunk
        )
        if layer == 1:
            max_id = chunk_id | self.get_segment_id_limit(chunk_id)

        return self.client.read_nodes(
            start_id=self.get_node_id(np.uint64(0), chunk_id=chunk_id),
            end_id=max_id,
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

    def get_atomic_ids_from_coords(
        self,
        coordinates: typing.Sequence[typing.Sequence[int]],
        parent_id: np.uint64,
        max_dist_nm: int = 150,
    ) -> typing.Sequence[np.uint64]:
        """Retrieves supervoxel ids for multiple coords.

        :param coordinates: n x 3 np.ndarray of locations in voxel space
        :param parent_id: parent id common to all coordinates at any layer
        :param max_dist_nm: max distance explored
        :return: supervoxel ids; returns None if no solution was found
        """
        if self.get_chunk_layer(parent_id) == 1:
            return np.array([parent_id] * len(coordinates), dtype=np.uint64)

        # Enable search with old parent by using its timestamp and map to parents
        parent_ts = self.get_node_timestamps([parent_id], return_numpy=False)[0]
        return id_helpers.get_atomic_ids_from_coords(
            self.meta,
            coordinates,
            parent_id,
            self.get_chunk_layer(parent_id),
            parent_ts,
            self.get_roots,
            max_dist_nm,
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
                    except KeyError as exc:
                        if fail_to_zero:
                            parents.append(0)
                        else:
                            exc.add_note(f"timestamp: {time_stamp}")
                            raise KeyError from exc
                parents = np.array(parents, dtype=basetypes.NODE_ID)
            else:
                for id_ in node_ids:
                    try:
                        parents.append(
                            [(p.value, p.timestamp) for p in parent_rows[id_]]
                        )
                    except KeyError as exc:
                        if fail_to_zero:
                            parents.append([(0, datetime.datetime.fromtimestamp(0))])
                        else:
                            raise KeyError from exc
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
            return np.concatenate(list(node_children_d.values()))
        return node_children_d

    def _get_children_multiple(
        self, node_ids: typing.Iterable[np.uint64], *, raw_only=False
    ) -> typing.Dict:
        if raw_only or not self.cache:
            node_children_d = self.client.read_nodes(
                node_ids=node_ids, properties=attributes.Hierarchy.Child
            )
            return {
                x: (
                    node_children_d[x][0].value
                    if x in node_children_d
                    else types.empty_1d.copy()
                )
                for x in node_ids
            }
        return self.cache.children_multiple(node_ids)

    def get_atomic_cross_edges(self, l2_ids: typing.Iterable) -> typing.Dict:
        """
        Returns atomic cross edges for level 2 IDs.
        A dict of the form `{l2id: {layer: atomic_cross_edges}}`.
        """
        node_edges_d_d = self.client.read_nodes(
            node_ids=l2_ids,
            properties=[
                attributes.Connectivity.AtomicCrossChunkEdge[l]
                for l in range(2, max(3, self.meta.layer_count))
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

    def get_cross_chunk_edges(
        self,
        node_ids: typing.Iterable,
        *,
        raw_only=False,
        all_layers=True,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Dict:
        """
        Returns cross edges for `node_ids`.
        A dict of the form `{node_id: {layer: cross_edges}}`.
        """
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)
        if raw_only or not self.cache:
            result = {}
            node_ids = np.array(node_ids, dtype=basetypes.NODE_ID)
            if node_ids.size == 0:
                return result
            layers = range(2, max(3, self.meta.layer_count))
            attrs = [attributes.Connectivity.CrossChunkEdge[l] for l in layers]
            node_edges_d_d = self.client.read_nodes(
                node_ids=node_ids,
                properties=attrs,
                end_time=time_stamp,
                end_time_inclusive=True,
            )
            layers = self.get_chunk_layers(node_ids)
            valid_layer = lambda x, y: x >= y
            if not all_layers:
                valid_layer = lambda x, y: x == y
            for layer, id_ in zip(layers, node_ids):
                try:
                    result[id_] = {
                        prop.index: val[0].value.copy()
                        for prop, val in node_edges_d_d[id_].items()
                        if valid_layer(prop.index, layer)
                    }
                except KeyError:
                    result[id_] = {}
            return result
        return self.cache.cross_chunk_edges_multiple(node_ids, time_stamp=time_stamp)

    def get_roots(
        self,
        node_ids: typing.Sequence[np.uint64],
        *,
        assert_roots: bool = False,
        time_stamp: typing.Optional[datetime.datetime] = None,
        stop_layer: int = None,
        ceil: bool = True,
        fail_to_zero: bool = False,
        raw_only=False,
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
                    unique_ids,
                    time_stamp=time_stamp,
                    fail_to_zero=fail_to_zero,
                    raw_only=raw_only,
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
        raw_only: bool = False,
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
                temp_parent_id = self.get_parent(
                    parent_id, time_stamp=time_stamp, raw_only=raw_only
                )
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
        """Determines whether root ids are superseded."""
        time_stamp = misc_utils.get_valid_timestamp(time_stamp)

        row_dict = self.client.read_nodes(
            node_ids=root_ids,
            properties=[attributes.Hierarchy.Child, attributes.Hierarchy.NewParent],
            end_time=time_stamp,
        )

        if len(row_dict) == 0:
            return np.zeros(len(root_ids), dtype=bool)

        latest_roots = [
            k for k, v in row_dict.items() if not attributes.Hierarchy.NewParent in v
        ]
        return np.isin(root_ids, latest_roots)

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
        return_layers: typing.List = None,
        nodes_only: bool = False,
        edges_only: bool = False,
        leaves_only: bool = False,
        return_flattened: bool = False,
    ) -> typing.Tuple[typing.Dict, typing.Tuple[Edges]]:
        """
        Generic subgraph method.
        """

        if return_layers is None:
            return_layers = [2]

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
        return_layers: typing.List = None,
        serializable: bool = False,
        return_flattened: bool = False,
    ) -> typing.Tuple[typing.Dict, typing.Dict, Edges]:
        """
        Get the children of `node_ids` that are at each of
        return_layers within the specified bounding box.
        """
        if return_layers is None:
            return_layers = [2]

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
        return get_subgraph_edges_and_leaves(
            self, node_id_or_ids, bbox, bbox_is_coordinate, False, True
        )

    def get_edited_edges(
        self, chunk_ids: np.ndarray, time_stamp: datetime.datetime = None
    ) -> typing.Dict:
        """
        Edges stored within a pcg that were created as a result of edits.
        Either 'fake' edges that were adding for a merge edit;
        Or 'split' edges resulting from a supervoxel split.
        """
        result = {}
        properties = [
            attributes.Connectivity.FakeEdges,
            attributes.Connectivity.SplitEdges,
            attributes.Connectivity.Affinity,
            attributes.Connectivity.Area,
        ]
        _edges_d = self.client.read_nodes(
            node_ids=chunk_ids,
            properties=properties,
            end_time=time_stamp,
            end_time_inclusive=True,
            fake_edges=True,
        )
        for id_, val in _edges_d.items():
            edges = val.get(attributes.Connectivity.FakeEdges, [])
            edges = np.concatenate([types.empty_2d, *[e.value for e in edges]])
            fake_edges_ = Edges(edges[:, 0], edges[:, 1])

            edges = val.get(attributes.Connectivity.SplitEdges, [])
            edges = np.concatenate([types.empty_2d, *[e.value for e in edges]])

            aff = val.get(attributes.Connectivity.Affinity, [])
            aff = np.concatenate([types.empty_affinities, *[e.value for e in aff]])

            areas = val.get(attributes.Connectivity.Area, [])
            areas = np.concatenate([types.empty_areas, *[e.value for e in areas]])
            split_edges_ = Edges(edges[:, 0], edges[:, 1], affinities=aff, areas=areas)

            result[id_] = fake_edges_ + split_edges_
        return result

    def copy_fake_edges(self, chunk_id: np.uint64) -> None:
        _edges = self.client.read_node(
            node_id=chunk_id,
            properties=attributes.Connectivity.FakeEdgesCF3,
            end_time_inclusive=True,
            fake_edges=True,
        )
        mutations = []
        _id = serializers.serialize_uint64(chunk_id, fake_edges=True)
        for e in _edges:
            val_dict = {attributes.Connectivity.FakeEdges: e.value}
            row = self.client.mutate_row(_id, val_dict, time_stamp=e.timestamp)
            mutations.append(row)
        self.client.write(mutations)

    def get_l2_agglomerations(
        self,
        level2_ids: np.ndarray,
        edges_only: bool = False,
        active: bool = False,
        time_stamp: typing.Optional[datetime.datetime] = None,
    ) -> typing.Tuple[typing.Dict[int, types.Agglomeration], typing.Tuple[Edges]]:
        """
        Children of Level 2 Node IDs and edges.
        Edges are read from cloud storage.
        """
        from .misc import get_agglomerations

        chunk_ids = np.unique(self.get_chunk_ids_from_node_ids(level2_ids))
        # google does not provide a storage emulator at the moment
        # this is an ugly hack to avoid permission issues in tests
        # find a better way to test
        edges_d = {}
        if self.mock_edges is None:
            edges_d = self.read_chunk_edges(chunk_ids)

        edited_edges = self.get_edited_edges(chunk_ids)
        all_chunk_edges = reduce(
            lambda x, y: x + y,
            chain(edges_d.values(), edited_edges.values()),
            Edges([], []),
        )
        if self.mock_edges is not None:
            all_chunk_edges += self.mock_edges

        if edges_only:
            if self.mock_edges is not None:
                all_chunk_edges = self.mock_edges.get_pairs()
            else:
                all_chunk_edges = all_chunk_edges.get_pairs()
            supervoxels = self.get_children(level2_ids, flatten=True)
            mask0 = np.in1d(all_chunk_edges[:, 0], supervoxels)
            mask1 = np.in1d(all_chunk_edges[:, 1], supervoxels)
            return all_chunk_edges[mask0 & mask1]

        l2id_children_d = self.get_children(level2_ids)
        sv_parent_d = {}
        for l2id in l2id_children_d:
            svs = l2id_children_d[l2id]
            for sv in svs:
                if sv in sv_parent_d:
                    raise ValueError("Found conflicting parents.")
            sv_parent_d.update(dict(zip(svs.tolist(), [l2id] * len(svs))))

        if active:
            all_chunk_edges = edge_utils.filter_inactive_cross_edges(
                self, all_chunk_edges, time_stamp=time_stamp
            )

        in_edges, out_edges, cross_edges = edge_utils.categorize_edges_v2(
            self.meta, all_chunk_edges, sv_parent_d
        )

        agglomeration_d = get_agglomerations(
            l2id_children_d, in_edges, out_edges, cross_edges, sv_parent_d
        )
        return (
            agglomeration_d,
            (
                (self.mock_edges,)
                if self.mock_edges is not None
                else (in_edges, out_edges, cross_edges)
            ),
        )

    def get_node_timestamps(
        self, node_ids: typing.Sequence[np.uint64], return_numpy=True, normalize=False
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
        result = []
        earliest_ts = self.get_earliest_timestamp()
        for n in node_ids:
            ts = children[n][0].timestamp
            if normalize:
                ts = earliest_ts if ts < earliest_ts else ts
            result.append(ts)
        if return_numpy:
            return np.array(result, dtype=np.datetime64)
        return result

    # OPERATIONS
    def add_edges(
        self,
        user_id: str,
        atomic_edges: typing.Sequence[typing.Sequence[np.uint64]],
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

    def get_chunk_coordinates_multiple(self, node_or_chunk_ids: typing.Sequence):
        node_or_chunk_ids = np.array(
            node_or_chunk_ids, dtype=basetypes.NODE_ID, copy=False
        )
        layers = self.get_chunk_layers(node_or_chunk_ids)
        assert np.all(layers == layers[0]), "All IDs must have the same layer."
        return chunk_utils.get_chunk_coordinates_multiple(self.meta, node_or_chunk_ids)

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

    def get_parent_chunk_id_multiple(self, node_or_chunk_ids: typing.Sequence):
        return chunk_hierarchy.get_parent_chunk_id_multiple(
            self.meta, node_or_chunk_ids
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
            self.get_chunk_coordinates_multiple(chunk_ids),
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

    def get_operation_ids(self, node_ids: typing.Sequence):
        response = self.client.read_nodes(node_ids=node_ids)
        result = {}
        for node in node_ids:
            try:
                operations = response[node][attributes.OperationLogs.OperationID]
                result[node] = [(x.value, x.timestamp) for x in operations]
            except KeyError:
                ...
        return result

    def get_single_leaf_multiple(self, node_ids):
        """Returns the first supervoxel found for each node_id."""
        result = {}
        node_ids_copy = np.copy(node_ids)
        children = np.copy(node_ids)
        children_d = self.get_children(node_ids)
        while True:
            children = [children_d[k][0] for k in children]
            children = np.array(children, dtype=basetypes.NODE_ID)
            mask = self.get_chunk_layers(children) == 1
            result.update(
                [(node, sv) for node, sv in zip(node_ids[mask], children[mask])]
            )
            node_ids = node_ids[~mask]
            children = children[~mask]
            if children.size == 0:
                break
            children_d = self.get_children(children)
        return np.array([result[k] for k in node_ids_copy], dtype=basetypes.NODE_ID)

    def get_chunk_layers_and_coordinates(self, node_or_chunk_ids: typing.Sequence):
        """
        Helper function that wraps get chunk layer and coordinates for nodes at any layer.
        """
        node_or_chunk_ids = np.array(node_or_chunk_ids, dtype=basetypes.NODE_ID)
        layers = self.get_chunk_layers(node_or_chunk_ids)
        chunk_coords = np.zeros(shape=(len(node_or_chunk_ids), 3), dtype=int)
        for _layer in np.unique(layers):
            mask = layers == _layer
            _nodes = node_or_chunk_ids[mask]
            chunk_coords[mask] = chunk_utils.get_chunk_coordinates_multiple(
                self.meta, _nodes
            )
        return layers, chunk_coords
