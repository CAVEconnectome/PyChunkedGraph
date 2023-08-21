# pylint: disable=invalid-name, missing-docstring

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Sequence
from typing import Optional

import numpy as np

from .edges import Edges
from .chunks.utils import normalize_bounding_box


class SubgraphProgress:
    """
    Helper class to keep track of node relationships
    while calling cg.get_subgraph(node_ids)
    """

    def __init__(self, meta, node_ids, return_layers, serializable):
        from collections import defaultdict

        self.meta = meta
        self.node_ids = node_ids
        self.return_layers = return_layers
        self.serializable = serializable

        self.node_to_subgraph = defaultdict(lambda: defaultdict(list))
        # "Frontier" of nodes that cg.get_children will be called on
        self.cur_nodes = np.array(list(node_ids), dtype=np.uint64)
        # Mapping of current frontier to self.node_ids
        self.cur_nodes_to_original_nodes = dict(zip(self.cur_nodes, self.cur_nodes))
        self.stop_layer = max(1, min(return_layers))
        self.create_initial_node_to_subgraph()

    def done_processing(self):
        return self.cur_nodes is None or len(self.cur_nodes) == 0

    def create_initial_node_to_subgraph(self):
        """
        Create initial subgraph. We will incrementally populate after processing
        each batch of children, and return it when there are no more to process.
        """
        from .chunks.utils import get_chunk_layer

        for node_id in self.cur_nodes:
            node_key = self.get_dict_key(node_id)
            node_layer = get_chunk_layer(self.meta, node_id)
            if node_layer in self.return_layers:
                self.node_to_subgraph[node_key][node_layer].append([node_id])

    def get_dict_key(self, node_id):
        if self.serializable:
            return str(node_id)
        return node_id

    def process_batch_of_children(self, cur_nodes_children):
        """
        Given children of self.cur_nodes, update subgraph and
        produce next frontier (if any).
        """
        from .chunks.utils import get_chunk_layers

        next_nodes_to_process = []
        next_nodes_to_original_nodes_keys = []
        next_nodes_to_original_nodes_values = []
        for cur_node, children in cur_nodes_children.items():
            children_layers = get_chunk_layers(self.meta, children)
            continue_mask = children_layers > self.stop_layer
            continue_children = children[continue_mask]
            original_id = self.cur_nodes_to_original_nodes[np.uint64(cur_node)]
            if len(continue_children) > 0:
                # These nodes will be in next frontier
                next_nodes_to_process.append(continue_children)
                next_nodes_to_original_nodes_keys.append(continue_children)
                next_nodes_to_original_nodes_values.append(
                    [original_id] * len(continue_children)
                )
            for return_layer in self.return_layers:
                # Update subgraph for each return_layer
                children_at_layer = children[children_layers == return_layer]
                if len(children_at_layer) > 0:
                    self.node_to_subgraph[self.get_dict_key(original_id)][
                        return_layer
                    ].append(children_at_layer)

        if len(next_nodes_to_process) == 0:
            self.cur_nodes = None
            # We are done, so we can np.concatenate/flatten each entry in node_to_subgraph
            self.flatten_subgraph()
        else:
            self.cur_nodes = np.concatenate(next_nodes_to_process)
            self.cur_nodes_to_original_nodes = dict(
                zip(
                    np.concatenate(next_nodes_to_original_nodes_keys),
                    np.concatenate(next_nodes_to_original_nodes_values),
                )
            )

    def flatten_subgraph(self):
        from .types import empty_1d

        # Flatten each entry in node_to_subgraph before returning
        for node_id in self.node_ids:
            for return_layer in self.return_layers:
                node_key = self.get_dict_key(node_id)
                children_at_layer = self.node_to_subgraph[node_key][return_layer]
                if len(children_at_layer) > 0:
                    self.node_to_subgraph[node_key][return_layer] = np.concatenate(
                        children_at_layer
                    )
                else:
                    self.node_to_subgraph[node_key][return_layer] = empty_1d


def get_subgraph_nodes(
    cg,
    node_id_or_ids: Union[np.uint64, Iterable],
    bbox: Optional[Sequence[Sequence[int]]] = None,
    bbox_is_coordinate: bool = False,
    return_layers: List = None,
    serializable: bool = False,
    return_flattened: bool = False,
) -> Tuple[Dict, Dict, Edges]:
    if return_layers is None:
        return_layers = [2]
    single = False
    node_ids = node_id_or_ids
    bbox = normalize_bounding_box(cg.meta, bbox, bbox_is_coordinate)
    if isinstance(node_id_or_ids, np.uint64) or isinstance(node_id_or_ids, int):
        single = True
        node_ids = [node_id_or_ids]
    layer_nodes_d = _get_subgraph_multiple_nodes(
        cg,
        node_ids=node_ids,
        bounding_box=bbox,
        return_layers=return_layers,
        serializable=serializable,
        return_flattened=return_flattened,
    )
    if single:
        if serializable:
            return layer_nodes_d[str(node_id_or_ids)]
        return layer_nodes_d[node_id_or_ids]
    return layer_nodes_d


def get_subgraph_edges_and_leaves(
    cg,
    node_id_or_ids: Union[np.uint64, Iterable],
    bbox: Optional[Sequence[Sequence[int]]] = None,
    bbox_is_coordinate: bool = False,
    edges_only: bool = False,
    leaves_only: bool = False,
) -> Tuple[Dict, Dict, Edges]:
    """Get the edges and/or leaves of the specified node_ids within the specified bounding box."""
    from .types import empty_1d

    node_ids = node_id_or_ids
    bbox = normalize_bounding_box(cg.meta, bbox, bbox_is_coordinate)
    if isinstance(node_id_or_ids, np.uint64) or isinstance(node_id_or_ids, int):
        node_ids = [node_id_or_ids]
    layer_nodes_d = _get_subgraph_multiple_nodes(
        cg, node_ids, bbox, return_layers=[2], return_flattened=True
    )
    level2_ids = [empty_1d]
    for node_id in node_ids:
        level2_ids.append(layer_nodes_d[node_id])
    level2_ids = np.concatenate(level2_ids)
    if leaves_only:
        return cg.get_children(level2_ids, flatten=True)
    if edges_only:
        return cg.get_l2_agglomerations(level2_ids, edges_only=True)
    return cg.get_l2_agglomerations(level2_ids)


def _get_subgraph_multiple_nodes(
    cg,
    node_ids: Iterable[np.uint64],
    bounding_box: Optional[Sequence[Sequence[int]]],
    return_layers: Sequence[int],
    serializable: bool = False,
    return_flattened: bool = False,
):
    from collections import ChainMap
    from multiwrapper.multiprocessing_utils import n_cpus
    from multiwrapper.multiprocessing_utils import multithread_func

    from .utils.generic import mask_nodes_by_bounding_box

    assert len(return_layers) > 0

    def _get_dict_key(raw_key):
        if serializable:
            return str(raw_key)
        return raw_key

    def _get_subgraph_multiple_nodes_threaded(
        node_ids_batch: Iterable[np.uint64],
    ) -> List[np.uint64]:
        children = cg.get_children(node_ids_batch)
        if bounding_box is not None:
            filtered_children = {}
            for node_id, nodes_children in children.items():
                if cg.get_chunk_layer(node_id) == 2:
                    # All children will be in same chunk so no need to check
                    filtered_children[_get_dict_key(node_id)] = nodes_children
                elif len(nodes_children) > 0:
                    bound_check_mask = mask_nodes_by_bounding_box(
                        cg.meta, nodes_children, bounding_box
                    )
                    filtered_children[_get_dict_key(node_id)] = nodes_children[
                        bound_check_mask
                    ]
            return filtered_children
        return children

    if bounding_box is not None:
        bounding_box = np.array(bounding_box)

    subgraph = SubgraphProgress(cg.meta, node_ids, return_layers, serializable)
    while not subgraph.done_processing():
        this_n_threads = min([int(len(subgraph.cur_nodes) // 50000) + 1, n_cpus])
        cur_nodes_child_maps = multithread_func(
            _get_subgraph_multiple_nodes_threaded,
            np.array_split(subgraph.cur_nodes, this_n_threads),
            n_threads=this_n_threads,
            debug=this_n_threads == 1,
        )
        cur_nodes_children = dict(ChainMap(*cur_nodes_child_maps))
        subgraph.process_batch_of_children(cur_nodes_children)

    if return_flattened and len(return_layers) == 1:
        for node_id in node_ids:
            subgraph.node_to_subgraph[
                _get_dict_key(node_id)
            ] = subgraph.node_to_subgraph[_get_dict_key(node_id)][return_layers[0]]

    return subgraph.node_to_subgraph
