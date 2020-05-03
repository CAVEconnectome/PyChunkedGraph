import numpy as np
import collections
from typing import Tuple

import cloudvolume
from google.cloud import bigtable

from . import ClusterIngestConfig
from . import IngestConfig
from .manager import IngestionManager
from ..graph import chunkedgraph
from ..graph.meta import ChunkedGraphMeta
from ..graph.meta import DataSource
from ..graph.meta import GraphConfig
from ..graph.meta import BigTableConfig
from ..graph.meta import BackendClientInfo

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def bootstrap(
    graph_id: str, config: dict, overwrite: bool = False, raw: bool = False
) -> Tuple[ChunkedGraphMeta, IngestConfig, BackendClientInfo]:
    """Parse config loaded from a yaml file."""
    ingest_config = IngestConfig(
        **config["ingest_config"],
        CLUSTER=ClusterIngestConfig(FLUSH_REDIS=True),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
    )
    bigtable_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(config["backend_client"]["TYPE"], bigtable_config)

    graph_config = GraphConfig(
        ID=f"{graph_id}", OVERWRITE=overwrite, **config["graph_config"],
    )
    data_source = DataSource(**config["data_source"])

    meta = ChunkedGraphMeta(graph_config, data_source)
    return (meta, ingest_config, client_info)


def postprocess_edge_data(im, edge_dict):
    data_version = im.chunkedgraph_meta.data_source.DATA_VERSION
    if data_version == 2:
        return edge_dict
    elif data_version in [3, 4]:
        new_edge_dict = {}
        for k in edge_dict:
            areas = (
                edge_dict[k]["area_x"] * im.chunkedgraph_meta.resolution[0]
                + edge_dict[k]["area_y"] * im.chunkedgraph_meta.resolution[1]
                + edge_dict[k]["area_z"] * im.chunkedgraph_meta.resolution[2]
            )

            affs = (
                edge_dict[k]["aff_x"] * im.chunkedgraph_meta.resolution[0]
                + edge_dict[k]["aff_y"] * im.chunkedgraph_meta.resolution[1]
                + edge_dict[k]["aff_z"] * im.chunkedgraph_meta.resolution[2]
            )

            new_edge_dict[k] = {}
            new_edge_dict[k]["sv1"] = edge_dict[k]["sv1"]
            new_edge_dict[k]["sv2"] = edge_dict[k]["sv2"]
            new_edge_dict[k]["area"] = areas
            new_edge_dict[k]["aff"] = affs

        return new_edge_dict
    else:
        raise Exception(f"Unknown data_version: {data_version}")
