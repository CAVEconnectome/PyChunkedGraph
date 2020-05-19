# def _test_l2_ids(self, node_l2ids_d):
#     # TODO remove, just for testing
#     node_ids = node_l2ids_d.keys()
#     node_all_l2ids_d = {}
#     for node_id in node_ids:
#         layer_nodes_d = self._get_subgraph_higher_layer_nodes(
#             node_id=node_id, bounding_box=None, return_layers=[2],
#         )
#         node_all_l2ids_d[node_id] = layer_nodes_d[2]
#     node_coords_d = {
#         node_id: self.get_chunk_coordinates(node_id) for node_id in node_ids
#     }
#     for node_id, (X, Y, Z) in node_coords_d.items():
#         chunks = chunk_utils.get_bounding_children_chunks(
#             self.meta, self.get_chunk_layer(node_id), (X, Y, Z), 2
#         )
#         bounding_chunk_ids = np.array(
#             [self.get_chunk_id(layer=2, x=x, y=y, z=z) for (x, y, z) in chunks],
#             dtype=basetypes.CHUNK_ID,
#         )
#         l2_chunk_ids = self.get_chunk_ids_from_node_ids(node_all_l2ids_d[node_id])
#         mask = np.in1d(l2_chunk_ids, bounding_chunk_ids)
#         bounding_l2_ids = node_all_l2ids_d[node_id][mask]
#         common = np.intersect1d(bounding_l2_ids, node_l2ids_d[node_id])
#         assert np.setdiff1d(bounding_l2_ids, common).size == 0
#         assert np.setdiff1d(node_l2ids_d[node_id], common).size == 0