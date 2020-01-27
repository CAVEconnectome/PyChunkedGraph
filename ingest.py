import sys
from multiprocessing import cpu_count

import numpy as np

from pychunkedgraph.ingest import IngestConfig
from pychunkedgraph.ingest.manager import IngestionManager
from pychunkedgraph.ingest.main import start_ingest
from pychunkedgraph.ingest.ingestion_utils import initialize_chunkedgraph
from pychunkedgraph.backend import DataSource
from pychunkedgraph.backend import GraphConfig
from pychunkedgraph.backend import BigTableConfig
from pychunkedgraph.backend import ChunkedGraphMeta


if __name__ == "__main__":
    name = sys.argv[1]
    ingest_config = IngestConfig()
    bigtable_config = BigTableConfig(table_id_prefix=name)

    graph_config = GraphConfig(
        graph_id=f"{bigtable_config.table_id_prefix}",
        chunk_size=np.array([512, 512, 128], dtype=int),
        overwrite=True,
    )

    data_source = DataSource(
        agglomeration="gs://ranl/scratch/pinky100_ca_com/agg",
        watershed="gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com",
        edges="gs://akhilesh-pcg/chunkedgraph/pinky100_sven/edges",
        components="gs://akhilesh-pcg/chunkedgraph/pinky100_sven/components",
        use_raw_edges=False,
        use_raw_components=False,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    initialize_chunkedgraph(meta)
    start_ingest(IngestionManager(ingest_config, meta))
