import numpy as np

from . import IngestConfig
from .manager import IngestionManager
from .ingestion import start_ingest
from .ingestion_utils import initialize_chunkedgraph
from ..backend import DataSource
from ..backend import GraphConfig
from ..backend import BigTableConfig
from ..backend import ChunkedGraphMeta


def ingest_graph():
    ingest_config = IngestConfig(build_graph=True)

    graph_config = GraphConfig(
        graph_id="ran-test",
        chunk_size=np.array([256, 256, 64], dtype=int),
        fanout=2,
        s_bits_atomic_layer=10,
    )
    bigtable_config = BigTableConfig(table_id_prefix="akhilesh-")

    bigtable_id = bigtable_config.table_id_prefix + graph_config.graph_id
    data_source = DataSource(
        agglomeration="gs://ranl/scratch/8da0f296103f6c8ff56c7097e9616406/agg",
        watershed="gs://neuroglancer/ranl/ws/8da0f296103f6c8ff56c7097e9616406",
        edges=f"gs://akhilesh-pcg/{bigtable_id}/edges",
        components=f"gs://akhilesh-pcg/{bigtable_id}/components",
        use_raw_edges=True,
        use_raw_components=True,
        data_version=4,
    )

    if ingest_config.build_graph:
        initialize_chunkedgraph(
            bigtable_id,
            data_source.watershed,
            graph_config.chunk_size,
            s_bits_atomic_layer=graph_config.s_bits_atomic_layer,
            edge_dir=data_source.edges,
        )

    meta = ChunkedGraphMeta(data_source, graph_config, bigtable_config)
    start_ingest(IngestionManager(ingest_config, meta))
