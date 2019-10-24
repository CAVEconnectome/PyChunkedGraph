## Ingestion

The following is a simple example to create a chunked graph on a single machine.

### Example

Run ingest with raw data. All the paths should be [CloudVolume](https://github.com/seung-lab/cloud-volume) compatible.
```
import numpy as np

from pychunkedgraph.ingest import IngestConfig
from pychunkedgraph.ingest.manager import IngestionManager
from pychunkedgraph.ingest.ingestion import start_ingest
from pychunkedgraph.ingest.ingestion_utils import initialize_chunkedgraph
from pychunkedgraph.backend import DataSource
from pychunkedgraph.backend import GraphConfig
from pychunkedgraph.backend import BigTableConfig
from pychunkedgraph.backend import ChunkedGraphMeta


ingest_config = IngestConfig(build_graph=True)

bigtable_config = BigTableConfig(table_id_prefix="prefix-")

graph_config = GraphConfig(
    graph_id=f"{bigtable_config.table_id_prefix}test-id",
    chunk_size=np.array([x, y, z], dtype=int),
    fanout=2,
    s_bits_atomic_layer=10, # TODO
)

data_source = DataSource(
    agglomeration="<path_to_agglomeration_data>",
    watershed="<path_to_watershed_data>",
    edges="<path_to_edges_data>",
    components="<path_to_components_data>",
    use_raw_edges=True,
    use_raw_components=True,
    data_version=4, # TODO
)

if ingest_config.build_graph:
    initialize_chunkedgraph(
        graph_config.graph_id,
        data_source.watershed,
        graph_config.chunk_size,
        s_bits_atomic_layer=graph_config.s_bits_atomic_layer,
    )

meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
start_ingest(IngestionManager(ingest_config, meta))
```

Raw data is processed and stored as edges and connected components per chunk for convenience.
Data stored in `DataSource.edges` and `DataSource.components` can be reused for building multiple chunkedgraphs for same dataset.
To skip this behavior, simply omit them.

If you already have edges and components stored per chunk then they can be used to build the chunkedgraph.
