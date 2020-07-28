from pychunkedgraph.app import app_utils
from pychunkedgraph.meshing import meshgen


def remeshing(table_id, lvl2_nodes):
    cg = app_utils.get_cg(table_id)

    # TODO: stop_layer and mip should be configurable by dataset
    meshgen.remeshing(
        cg, lvl2_nodes, stop_layer=4, cv_path=None,
        cv_mesh_dir=None, mip=1, max_err=320
    )
