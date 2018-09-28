import numpy as np

import pychunkedgraph.backend.key_utils

partner_key = 'atomic_partners'
partner_key_s = pychunkedgraph.backend.key_utils.serialize_key(partner_key)
affinity_key = 'affinities'
affinity_key_s = pychunkedgraph.backend.key_utils.serialize_key(affinity_key)
area_key = 'areas'
area_key_s = pychunkedgraph.backend.key_utils.serialize_key(area_key)
connected_key = 'connected'
connected_key_s = pychunkedgraph.backend.key_utils.serialize_key(connected_key)
disconnected_key = 'disconnected'
disconnected_key_s = pychunkedgraph.backend.key_utils.serialize_key(disconnected_key)
parent_key = 'parents'
parent_key_s = pychunkedgraph.backend.key_utils.serialize_key(parent_key)
cross_chunk_edge_keyformat = 'atomic_cross_edges_%d'


dtype_dict = {partner_key: np.uint64,
              partner_key_s: np.uint64,
              affinity_key: np.float32,
              affinity_key_s: np.float32,
              area_key: np.uint64,
              area_key_s: np.uint64,
              connected_key: np.uint64,
              connected_key_s: np.uint64,
              parent_key: np.int,
              parent_key_s: np.int,
              "cross_chunk_edges": np.uint64}
