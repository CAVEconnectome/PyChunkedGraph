from pychunkedgraph.app import app_utils
from pychunkedgraph.meshing import meshgen, meshgen_utils
import numpy as np
import os


def remeshing(table_id, lvl2_nodes):
    lvl2_nodes = np.array(lvl2_nodes, dtype=np.uint64)
    cg = app_utils.get_cg(table_id, skip_cache=True)

    cv_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(
        cg.meta.data_source.WATERSHED, cv_mesh_dir, cv_unsharded_mesh_dir
    )
    mesh_data = cg.meta.custom_data["mesh"]

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg,
        lvl2_nodes,
        stop_layer=mesh_data["max_layer"],
        mip=mesh_data["mip"],
        max_err=mesh_data["max_error"],
        cv_sharded_mesh_dir=cv_mesh_dir,
        cv_unsharded_mesh_path=cv_unsharded_mesh_path,
    )