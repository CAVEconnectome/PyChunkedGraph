## Ingestion

The following is a simple example to create a chunked graph on a single machine.

### Example

Run ingest with raw data.
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
    chunk_size=np.array([256, 256, 64], dtype=int),
    fanout=2,
    s_bits_atomic_layer=10,
)

data_source = DataSource(
    agglomeration="gs://ranl/scratch/8da0f296103f6c8ff56c7097e9616406/agg",
    watershed="gs://neuroglancer/ranl/ws/8da0f296103f6c8ff56c7097e9616406",
    edges=f"gs://<bucket>/{graph_config.graph_id}/edges",
    components=f"gs://<bucket>/{graph_config.graph_id}/components",
    use_raw_edges=True,
    use_raw_components=True,
    data_version=4,
)

if ingest_config.build_graph:
    initialize_chunkedgraph(
        graph_config.graph_id,
        data_source.watershed,
        graph_config.chunk_size,
        s_bits_atomic_layer=graph_config.s_bits_atomic_layer,
        edge_dir=data_source.edges,
    )

meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
start_ingest(IngestionManager(ingest_config, meta))
```

Raw data is processed and stored as edges and connected components per chunk for convenience.
Data stored in `DataSource.edges` and `DataSource.components` can be reused for building chunkedgraphs for same dataset.
