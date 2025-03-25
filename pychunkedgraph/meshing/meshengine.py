import cloudvolume
import numpy as np
import itertools
import random

from pychunkedgraph.backend import chunkedgraph
from multiwrapper import multiprocessing_utils as mu

from . import meshgen


class MeshEngine(object):
    def __init__(self,
                 table_id: str,
                 instance_id: str = "pychunkedgraph",
                 project_id: str = "neuromancer-seung-import",
                 mesh_mip: int = 3,
                 highest_mesh_layer: int = 5):

        self._table_id = table_id
        self._instance_id = instance_id
        self._project_id = project_id
        self._mesh_mip = mesh_mip
        self._highest_mesh_layer = highest_mesh_layer

        # Defaults for external object instances
        self._cv = None
        self._cg = None

    @property
    def table_id(self):
        return self._table_id

    @property
    def table_name(self):
        name_parts = self.table_id.split("_")[1:]
        table_name = name_parts[0]
        for name_part in name_parts[1:]:
            table_name += "_" + name_part

        return table_name

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def project_id(self):
        return self._project_id

    @property
    def mesh_mip(self):
        return self._mesh_mip

    @property
    def highest_mesh_layer(self):
        return self._highest_mesh_layer

    @property
    def cg(self):
        if self._cg is None:
            self._cg = chunkedgraph.ChunkedGraph(
                table_id=self.table_id,
                instance_id=self.instance_id,
                project_id=self.project_id)
        return self._cg

    @property
    def cv_mesh_dir(self):
        return "mesh_mip_%d_%s" % (self.mesh_mip, self.table_name)

    @property
    def cv_path(self):
        return self.cg._cv_path

    @property
    def cv(self):
        if self._cv is None:
            self._cv = cloudvolume.CloudVolume(self.cv_path)
            self._cv.info["mesh"] = self.cv_mesh_dir
        return self._cv

    def mesh_multiple_layers(self, layers=None, bounding_box=None,
                             block_factor=2, n_threads=128):
        if layers is None:
            layers = range(1, int(self.cg.n_layers + 1))

        layers = np.array(layers, dtype=int)

        layers = layers[layers > 0]
        layers = layers[layers < self.highest_mesh_layer + 1]

        print("Meshing layers:", layers)

        for layer in layers:
            print("Now: layer %d" % layer)
            self.mesh_single_layer(layer, bounding_box=bounding_box,
                                   block_factor=block_factor,
                                   n_threads=n_threads)

    def mesh_single_layer(self, layer, bounding_box=None, block_factor=2,
                          n_threads=128):
        assert layer <= self.highest_mesh_layer

        dataset_bounding_box = np.array(self.cv.bounds.to_list())

        block_bounding_box_cg = \
            [np.floor(dataset_bounding_box[:3] /
                      self.cg.chunk_size).astype(int),
             np.ceil(dataset_bounding_box[3:] /
                     self.cg.chunk_size).astype(int)]

        if bounding_box is not None:
            bounding_box_cg = \
                [np.floor(bounding_box[0] /
                          self.cg.chunk_size).astype(int),
                 np.ceil(bounding_box[1] /
                         self.cg.chunk_size).astype(int)]

            m = block_bounding_box_cg[0] < bounding_box_cg[0]
            block_bounding_box_cg[0][m] = bounding_box_cg[0][m]

            m = block_bounding_box_cg[1] > bounding_box_cg[1]
            block_bounding_box_cg[1][m] = bounding_box_cg[1][m]

        block_bounding_box_cg /= 2 ** np.max([0, layer - 2])
        block_bounding_box_cg = np.ceil(block_bounding_box_cg)

        n_jobs = np.product(block_bounding_box_cg[1] -
                            block_bounding_box_cg[0]) / \
                 block_factor ** 2 < n_threads

        while n_jobs < n_threads and block_factor > 1:
            block_factor -= 1

            n_jobs = np.product(block_bounding_box_cg[1] -
                                block_bounding_box_cg[0]) / \
                     block_factor ** 2 < n_threads

        block_iter = itertools.product(np.arange(block_bounding_box_cg[0][0],
                                                 block_bounding_box_cg[1][0],
                                                 block_factor),
                                       np.arange(block_bounding_box_cg[0][1],
                                                 block_bounding_box_cg[1][1],
                                                 block_factor),
                                       np.arange(block_bounding_box_cg[0][2],
                                                 block_bounding_box_cg[1][2],
                                                 block_factor))

        blocks = np.array(list(block_iter), dtype=int)

        cg_info = self.cg.get_serialized_info()
        del (cg_info['credentials'])

        multi_args = []
        for start_block in blocks:
            end_block = start_block + block_factor
            m = end_block > block_bounding_box_cg[1]
            end_block[m] = block_bounding_box_cg[1][m]

            multi_args.append([cg_info, start_block, end_block, self.cg._cv_path,
                               self.cv_mesh_dir, self.mesh_mip, layer])
        random.shuffle(multi_args)

        random.shuffle(multi_args)

        # Run parallelizing
        if n_threads == 1:
            mu.multiprocess_func(meshgen._mesh_layer_thread, multi_args,
                                 n_threads=n_threads, verbose=True,
                                 debug=n_threads == 1)
        else:
            mu.multisubprocess_func(meshgen._mesh_layer_thread, multi_args,
                                    n_threads=n_threads,
                                    suffix="%s_%d" % (self.table_id, layer))

    def create_manifests_for_higher_layers(self, n_threads=1):
        root_id_max = self.cg.get_max_node_id(
            self.cg.get_chunk_id(layer=int(self.cg.n_layers),
                                 x=int(0), y=int(0),
                                 z=int(0)))

        root_id_blocks = np.linspace(1, root_id_max, n_threads*3).astype(int)
        cg_info = self.cg.get_serialized_info()
        del (cg_info['credentials'])

        multi_args = []
        for i_block in range(len(root_id_blocks) - 1):
            multi_args.append([cg_info, self.cv_path, self.cv_mesh_dir,
                               root_id_blocks[i_block],
                               root_id_blocks[i_block + 1],
                               self.highest_mesh_layer])

        # Run parallelizing
        if n_threads == 1:
            mu.multiprocess_func(meshgen._create_manifest_files_thread,
                                 multi_args, n_threads=n_threads, verbose=True,
                                 debug=n_threads == 1)
        else:
            mu.multisubprocess_func(meshgen._create_manifest_files_thread,
                                    multi_args, n_threads=n_threads)
