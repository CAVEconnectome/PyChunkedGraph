import numpy as np
from . import basetypes


class SubgraphProgress:
    """
    Helper class to keep track of node relationships
    while calling cg.get_subgraph(node_ids)
    """

    def __init__(self, cg, node_ids, return_layers, serializable):
        self.node_ids = node_ids
        self.return_layers = return_layers
        self.cg = cg
        self.serializable = serializable

        self.node_to_subgraph = {}
        # "Frontier" of nodes that cg.get_children will be called on
        self.cur_nodes = np.array(list(node_ids), dtype=np.uint64)
        # Mapping of current frontier to self.node_ids
        self.cur_nodes_to_original_nodes = dict(
            zip(self.cur_nodes, self.cur_nodes)
        )
        self.stop_layer = max(1, np.min(return_layers))
        self.create_initial_node_to_subgraph()

    def done_processing(self):
        return self.cur_nodes is None or len(self.cur_nodes) == 0

    def create_initial_node_to_subgraph(self):
        """
        Create initial subgraph. We will incrementally populate after processing
        each batch of children, and return it when there are no more to process.
        """
        for node_id in self.cur_nodes:
            node_key = self.get_dict_key(node_id)
            self.node_to_subgraph[node_key] = {}
            for return_layer in self.return_layers:
                self.node_to_subgraph[node_key][return_layer] = []
            node_layer = self.cg.get_chunk_layer(node_id)
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
        next_nodes_to_process = []
        next_nodes_to_original_nodes_keys = []
        next_nodes_to_original_nodes_values = []
        for cur_node, children in cur_nodes_children.items():
            children_layers = self.cg.get_chunk_layers(children)
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
            # We are done, so we can concatenate/flatten each entry in node_to_subgraph
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
        # Flatten each entry in node_to_subgraph before returning
        for node_id in self.node_ids:
            for return_layer in self.return_layers:
                node_key = self.get_dict_key(node_id)
                children_at_layer = self.node_to_subgraph[node_key][
                    return_layer
                ]
                if len(children_at_layer) > 0:
                    self.node_to_subgraph[node_key][
                        return_layer
                    ] = np.concatenate(children_at_layer)
                else:
                    self.node_to_subgraph[node_key][return_layer] = np.empty(
                        0, dtype=basetypes.NODE_ID
                    )
