import itertools
import numpy as np

from pychunkedgraph.backend import chunkedgraph


class IngestionManager(object):
    def __init__(self, storage_path, cg_table_id=None, n_layers=None,
                 instance_id=None, project_id=None, data_version=2):
        self._storage_path = storage_path
        self._cg_table_id = cg_table_id
        self._instance_id = instance_id
        self._project_id = project_id
        self._cg = None
        self._n_layers = n_layers
        self._data_version = data_version

    @property
    def storage_path(self):
        return self._storage_path

    @property
    def data_version(self):
        assert self._data_version in [2, 3, 4]
        return self._data_version

    @property
    def edge_dtype(self):
        if self.data_version == 4:
            dtype = [("sv1", np.uint64), ("sv2", np.uint64),
                     ("aff_x", np.float32), ("area_x", np.uint64),
                     ("aff_y", np.float32), ("area_y", np.uint64),
                     ("aff_z", np.float32), ("area_z", np.uint64)]
        elif self.data_version == 3:
            dtype = [("sv1", np.uint64), ("sv2", np.uint64),
                     ("aff_x", np.float64), ("area_x", np.uint64),
                     ("aff_y", np.float64), ("area_y", np.uint64),
                     ("aff_z", np.float64), ("area_z", np.uint64)]
        elif self.data_version == 2:
            dtype = [("sv1", np.uint64), ("sv2", np.uint64),
                     ("aff", np.float32), ("area", np.uint64)]
        else:
            raise Exception()

        return dtype

    @property
    def cg(self):
        if self._cg is None:
            kwargs = {}

            if self._instance_id is not None:
                kwargs["instance_id"] = self._instance_id

            if self._project_id is not None:
                kwargs["project_id"] = self._project_id

            self._cg = chunkedgraph.ChunkedGraph(table_id=self._cg_table_id,
                                                 **kwargs)

        return self._cg

    @property
    def bounds(self):
        bounds = self.cg.vx_vol_bounds.copy()
        bounds -= self.cg.vx_vol_bounds[:, 0:1]

        return bounds

    @property
    def chunk_id_bounds(self):
        return np.ceil((self.bounds / self.cg.chunk_size[:, None])).astype(np.int)

    @property
    def chunk_coord_gen(self):
        return itertools.product(*[range(*r) for r in self.chunk_id_bounds])

    @property
    def chunk_coords(self):
        return np.array(list(self.chunk_coord_gen), dtype=np.int)

    @property
    def n_layers(self):
        if self._n_layers is None:
            self._n_layers = self.cg.n_layers
        return self._n_layers

    def get_serialized_info(self):
        info = {"storage_path": self.storage_path,
                "cg_table_id": self._cg_table_id,
                "n_layers": self.n_layers,
                "instance_id": self._instance_id,
                "project_id": self._project_id,
                "data_version": self.data_version}

        return info

    def is_out_of_bounce(self, chunk_coordinate):
        if np.any(chunk_coordinate < 0):
            return True

        if np.any(chunk_coordinate > 2**self.cg.bitmasks[1]):
            return True

        return False

