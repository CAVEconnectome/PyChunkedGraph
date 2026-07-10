from pychunkedgraph.app import app_utils
from pychunkedgraph.meshing import meshgen
from pychunkedgraph.meshing.mesh_meta import MeshMeta
import numpy as np


def remeshing(table_id, lvl2_nodes):
    lvl2_nodes = np.array(lvl2_nodes, dtype=np.uint64)
    cg = app_utils.get_cg(table_id, skip_cache=True)
    mm = MeshMeta(cg)
    meshgen.remeshing(
        cg,
        lvl2_nodes,
        stop_layer=mm.max_layer,
        mip=mm.mip,
        max_err=mm.max_error,
        cv_sharded_mesh_dir=mm.dir,
        cv_unsharded_mesh_path=mm.dynamic_path,
    )