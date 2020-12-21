from taskqueue import RegisteredTask
from pychunkedgraph.meshing import meshgen
import numpy as np


class MeshTask(RegisteredTask):
    def __init__(self, cg_name, layer, chunk_id, mip, cv_mesh_dir, cache=True):
        super().__init__(cg_name, layer, chunk_id, mip, cv_mesh_dir, cache)

    def execute(self):
        cg_name = self.cg_name
        chunk_id = np.uint64(self.chunk_id)
        cv_mesh_dir = self.cv_mesh_dir
        mip = self.mip
        layer = self.layer
        if layer == 2:
            result = meshgen.chunk_initial_mesh_task(
                cg_name,
                chunk_id,
                None,
                mip=mip,
                sharded=True,
                cv_mesh_dir=cv_mesh_dir,
                cache=self.cache
            )
        else:
            result = meshgen.chunk_initial_sharded_stitching_task(
                cg_name, chunk_id, mip, cv_mesh_dir, cache=self.cache
            )
        print(result)

