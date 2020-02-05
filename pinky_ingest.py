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
    count = int(sys.argv[2])
    ingest_config = IngestConfig(build_graph=True)
    bigtable_config = BigTableConfig(table_id_prefix=name)

    graph_config = GraphConfig(
        graph_id=f"{bigtable_config.table_id_prefix}",
        chunk_size=np.array([512, 512, 128], dtype=int),
        overwrite=True,
    )

    data_source = DataSource(
        agglomeration="gs://ranl/scratch/1638876bcc1a25b55688bed837db6e73/agg",
        watershed="gs://neuroglancer/pinky100_v0/ws/pinky100_ca_com",
        edges="gs://akhilesh-pcg/1638876bcc1a25b55688bed837db6e73/edges",
        components="gs://akhilesh-pcg/1638876bcc1a25b55688bed837db6e73/components",
        use_raw_edges=True,
        use_raw_components=True,
    )

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    if ingest_config.build_graph:
        initialize_chunkedgraph(meta)
    start_ingest(IngestionManager(ingest_config, meta), n_workers=count)
