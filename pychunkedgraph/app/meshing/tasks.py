from pychunkedgraph.app import app_utils
from pychunkedgraph.meshing import meshgen
import numpy as np
from flask import current_app


def remeshing(table_id, lvl2_nodes):
    lvl2_nodes = np.array(lvl2_nodes, dtype=np.uint64)
    cg = app_utils.get_cg(table_id)
    
    current_app.logger.debug(f"remeshing {lvl2_nodes} {cg.get_serialized_info()}")

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg, lvl2_nodes, stop_layer=4, cv_path=None,
        cv_mesh_dir=None, mip=1, max_err=320
    )
