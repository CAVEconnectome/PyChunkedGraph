import numpy as np
import collections

import cloudvolume
from google.cloud import bigtable

from pychunkedgraph.backend import ChunkedGraphMeta
from pychunkedgraph.backend import BigTableConfig
from pychunkedgraph.backend import chunkedgraph
from pychunkedgraph.backend import chunkedgraph_utils

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def _table_exists(bigtable_config: BigTableConfig, table_id: str):
    client = bigtable.Client(project=bigtable_config.project_id, admin=True)
    instance = client.instance(bigtable_config.instance_id)
    table = instance.table(table_id)
    return table.exists()


def initialize_chunkedgraph(
    meta: ChunkedGraphMeta, cg_mesh_dir="mesh_dir", n_bits_root_counter=8, size=None
):
    """ Initalizes a chunkedgraph on BigTable """
    if not meta.graph_config.overwrite and _table_exists(
        meta.bigtable_config, meta.graph_config.graph_id
    ):
        raise ValueError(f"{meta.graph_config.graph_id} already exists.")

    ws_cv = cloudvolume.CloudVolume(meta.data_source.watershed)
    if size is not None:
        size = np.array(size)
        for i in range(len(ws_cv.info["scales"])):
            original_size = ws_cv.info["scales"][i]["size"]
            size = np.min([size, original_size], axis=0)
            ws_cv.info["scales"][i]["size"] = [int(x) for x in size]
            size[:-1] //= 2

    dataset_info = ws_cv.info
    dataset_info["mesh"] = cg_mesh_dir
    dataset_info["data_dir"] = meta.data_source.watershed
    dataset_info["graph"] = {
        "chunk_size": [int(s) for s in meta.graph_config.chunk_size]
    }

    kwargs = {
        "instance_id": meta.bigtable_config.instance_id,
        "project_id": meta.bigtable_config.project_id,
        "table_id": meta.graph_config.graph_id,
        "chunk_size": meta.graph_config.chunk_size,
        "fan_out": np.uint64(meta.graph_config.fanout),
        "n_layers": np.uint64(meta.layer_count),
        "dataset_info": dataset_info,
        "use_skip_connections": meta.graph_config.use_skip_connections,
        "s_bits_atomic_layer": meta.graph_config.s_bits_atomic_layer,
        "n_bits_root_counter": n_bits_root_counter,
        "is_new": True,
    }
    return chunkedgraph.ChunkedGraph(**kwargs)


def postprocess_edge_data(im, edge_dict):
    data_version = im.chunkedgraph_meta.data_source.data_version
    if data_version == 2:
        return edge_dict
    elif data_version in [3, 4]:
        new_edge_dict = {}
        for k in edge_dict:
            areas = (
                edge_dict[k]["area_x"] * im.cg.cv.resolution[0]
                + edge_dict[k]["area_y"] * im.cg.cv.resolution[1]
                + edge_dict[k]["area_z"] * im.cg.cv.resolution[2]
            )

            affs = (
                edge_dict[k]["aff_x"] * im.cg.cv.resolution[0]
                + edge_dict[k]["aff_y"] * im.cg.cv.resolution[1]
                + edge_dict[k]["aff_z"] * im.cg.cv.resolution[2]
            )

            new_edge_dict[k] = {}
            new_edge_dict[k]["sv1"] = edge_dict[k]["sv1"]
            new_edge_dict[k]["sv2"] = edge_dict[k]["sv2"]
            new_edge_dict[k]["area"] = areas
            new_edge_dict[k]["aff"] = affs

        return new_edge_dict
    else:
        raise Exception(f"Unknown data_version: {data_version}")
