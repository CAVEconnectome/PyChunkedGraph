from taskqueue import RegisteredTask
from pychunkedgraph.meshing import meshgen
import numpy as np
import pickle


class MeshTask(RegisteredTask):
    def __init__(self, cg_name, chunk_id, mip, cv_graphene_path, cv_mesh_dir):
        super().__init__(cg_name, chunk_id, mip, cv_graphene_path, cv_mesh_dir)

    def execute(self):
        cg_name = self.cg_name
        chunk_id = np.uint64(self.chunk_id)
        cv_graphene_path = self.cv_graphene_path
        cv_mesh_dir = self.cv_mesh_dir
        mip = self.mip
        blah = meshgen.chunk_initial_sharded_stitching_task(
            cg_name, chunk_id, mip, cv_graphene_path, cv_mesh_dir
        )
        print(pickle.loads(blah))

