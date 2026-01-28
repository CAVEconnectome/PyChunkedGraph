from typing import Tuple


from . import ClusterIngestConfig
from . import IngestConfig
from ..graph.meta import ChunkedGraphMeta
from ..graph.meta import DataSource
from ..graph.meta import GraphConfig

from ..graph.client import BackendClientInfo
from ..graph.client.bigtable import BigTableConfig
from ..graph.client.hbase import HBaseConfig

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def bootstrap(
    graph_id: str,
    config: dict,
    overwrite: bool = False,
    raw: bool = False,
    test_run: bool = False,
) -> Tuple[ChunkedGraphMeta, IngestConfig, BackendClientInfo]:
    """Parse config loaded from a yaml file."""
    ingest_config = IngestConfig(
        **config.get("ingest_config", {}),
        CLUSTER=ClusterIngestConfig(),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
        TEST_RUN=test_run,
    )
    backend_type = config["backend_client"].get("TYPE", "bigtable")
    if backend_type.lower() == "hbase":
        client_config = HBaseConfig(**config["backend_client"]["CONFIG"])
    else:
        client_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(backend_type, client_config)

    graph_config = GraphConfig(
        ID=f"{graph_id}",
        OVERWRITE=overwrite,
        **config["graph_config"],
    )
    data_source = DataSource(**config["data_source"])

    meta = ChunkedGraphMeta(graph_config, data_source)
    return (meta, ingest_config, client_info)


def postprocess_edge_data(im, edge_dict):
    data_version = im.cg_meta.data_source.DATA_VERSION
    if data_version == 2:
        return edge_dict
    elif data_version in [3, 4]:
        new_edge_dict = {}
        for k in edge_dict:
            new_edge_dict[k] = {}
            if edge_dict[k] is None or len(edge_dict[k]) == 0:
                continue

            areas = (
                edge_dict[k]["area_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["area_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["area_z"] * im.cg_meta.resolution[2]
            )

            affs = (
                edge_dict[k]["aff_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["aff_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["aff_z"] * im.cg_meta.resolution[2]
            )

            new_edge_dict[k]["sv1"] = edge_dict[k]["sv1"]
            new_edge_dict[k]["sv2"] = edge_dict[k]["sv2"]
            new_edge_dict[k]["area"] = areas
            new_edge_dict[k]["aff"] = affs

        return new_edge_dict
    else:
        raise Exception(f"Unknown data_version: {data_version}")
