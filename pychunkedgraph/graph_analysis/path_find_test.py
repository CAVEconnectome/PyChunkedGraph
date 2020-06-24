from pychunkedgraph.graph.chunkedgraph import ChunkedGraph
from pychunkedgraph.graph_analysis import analysis
import numpy as np

cg = ChunkedGraph(graph_id='minnie3_v1')
sv_source = np.uint64(98112859459918426)
# sv_target = np.uint64(98253459442619149)
sv_target = np.uint64(98816134584789309)
source_id = cg.get_parent(sv_source)
target_id = cg.get_parent(sv_target)
# target_id = np.uint64(170873728622657590)
l2_path = analysis.find_l2_shortest_path(cg, source_id, target_id)
print(analysis.compute_mesh_centroids_of_l2_ids(cg, l2_path, flatten=True))