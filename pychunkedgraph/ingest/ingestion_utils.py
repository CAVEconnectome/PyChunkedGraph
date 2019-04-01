import numpy as np
from pychunkedgraph.backend import chunkedgraph, chunkedgraph_utils

import cloudvolume


def initialize_chunkedgraph(cg_table_id, ws_cv_path, chunk_size, cg_mesh_dir,
                            fan_out=2, instance_id=None, project_id=None):
    """ Initalizes a chunkedgraph on BigTable

    :param cg_table_id: str
        name of chunkedgraph
    :param ws_cv_path: str
        path to watershed segmentation on Google Cloud
    :param chunk_size: np.ndarray
        array of three ints
    :param cg_mesh_dir: str
        mesh folder name
    :param fan_out: int
        fan out of chunked graph (2 == Octree)
    :param instance_id: str
        Google instance id
    :param project_id: str
        Google project id
    :return: ChunkedGraph
    """
    ws_cv = cloudvolume.CloudVolume(ws_cv_path)
    bbox = np.array(ws_cv.bounds.to_list()).reshape(2, 3)

    # assert np.all(bbox[0] == 0)
    # assert np.all((bbox[1] % chunk_size) == 0)

    n_chunks = ((bbox[1] - bbox[0]) / chunk_size).astype(np.int)
    n_layers = int(np.ceil(chunkedgraph_utils.log_n(np.max(n_chunks), fan_out))) + 2

    dataset_info = ws_cv.info
    dataset_info["mesh"] = cg_mesh_dir
    dataset_info["data_dir"] = ws_cv_path
    dataset_info["graph"] = {"chunk_size": [int(s) for s in chunk_size]}

    kwargs = {"table_id": cg_table_id,
              "chunk_size": chunk_size,
              "fan_out": np.uint64(fan_out),
              "n_layers": np.uint64(n_layers),
              "dataset_info": dataset_info,
              "is_new": True}

    if instance_id is not None:
        kwargs["instance_id"] = instance_id

    if project_id is not None:
        kwargs["project_id"] = project_id

    cg = chunkedgraph.ChunkedGraph(**kwargs)

    return cg
