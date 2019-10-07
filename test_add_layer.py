import numpy as np

from pychunkedgraph.ingest import IngestConfig
from pychunkedgraph.ingest.ingestionmanager import IngestionManager
from pychunkedgraph.backend.definitions.config import DataSource
from pychunkedgraph.backend.definitions.config import GraphConfig
from pychunkedgraph.backend.definitions.config import BigTableConfig

from pychunkedgraph.backend.chunkedgraph import ChunkedGraph
from pychunkedgraph.ingest.initialization.abstract_layers import add_layer
from pychunkedgraph.ingest.ran_ingestion_v2 import enqueue_atomic_tasks
from pychunkedgraph.ingest.ran_ingestion_v2 import _get_children_coords

processed = True
graph_id = "akhilesh-minnie65"

ingest_config = IngestConfig(build_graph=True)
data_source = DataSource(
    agglomeration="gs://ranl-scratch/minnie65_0/agg",
    watershed="gs://microns-seunglab/minnie65/ws_minnie65_0",
    edges="gs://chunkedgraph/minnie65_0/edges",
    components="gs://chunkedgraph/minnie65_0/components",
    use_raw_edges=not processed,
    use_raw_components=not processed,
    data_version=2,
)
graph_config = GraphConfig(
    graph_id=graph_id,
    chunk_size=np.array([256, 256, 512], dtype=int),
    fanout=2,
    s_bits_atomic_layer=10,
)
bigtable_config = BigTableConfig()
imanager = IngestionManager(ingest_config, data_source, graph_config, bigtable_config)

cg = ChunkedGraph("akhilesh-minnie65")
layer = 7
parent_coords = [10, 6, 0]
children_coords = _get_children_coords(imanager, layer-1, parent_coords)

print(len(children_coords))
add_layer(cg, layer, parent_coords, children_coords)
