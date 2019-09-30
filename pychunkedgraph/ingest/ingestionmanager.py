import itertools
import numpy as np
import pickle


from ..backend.chunkedgraph_utils import compute_bitmasks
from ..backend.chunkedgraph import ChunkedGraph
from ..utils.redis import get_rq_queue


class IngestionManager(object):
    def __init__(
        self,
        storage_path,
        cg_table_id=None,
        n_layers=None,
        instance_id=None,
        project_id=None,
        cv=None,
        chunk_size=None,
        data_version=2,
        use_raw_edge_data=True,
        use_raw_agglomeration_data=True,
        edges_dir=None,
        components_dir=None,
        task_queue_name="test",
        build_graph=True,
    ):
        self._storage_path = storage_path
        self._cg_table_id = cg_table_id
        self._instance_id = instance_id
        self._project_id = project_id
        self._cg = None
        self._n_layers = n_layers
        self._data_version = data_version
        self._cv = cv
        self._chunk_size = chunk_size
        self._use_raw_edge_data = use_raw_edge_data
        self._use_raw_agglomeration_data = use_raw_agglomeration_data
        self._edges_dir = edges_dir
        self._components_dir = components_dir
        self._chunk_coords = None
        self._layer_bounds_d = None
        self._redis_connection = None
        self._task_q_name = task_queue_name
        self._task_q = None
        self._build_graph = True
        self._bitmasks = None
        self._bounds = None

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
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff_x", np.float32),
                ("area_x", np.uint64),
                ("aff_y", np.float32),
                ("area_y", np.uint64),
                ("aff_z", np.float32),
                ("area_z", np.uint64),
            ]
        elif self.data_version == 3:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff_x", np.float64),
                ("area_x", np.uint64),
                ("aff_y", np.float64),
                ("area_y", np.uint64),
                ("aff_z", np.float64),
                ("area_z", np.uint64),
            ]
        elif self.data_version == 2:
            dtype = [
                ("sv1", np.uint64),
                ("sv2", np.uint64),
                ("aff", np.float32),
                ("area", np.uint64),
            ]
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

            self._cg = ChunkedGraph(table_id=self._cg_table_id, **kwargs)

        return self._cg

    @property
    def bounds(self):
        if self._bounds:
            return self._bounds
        cv_bounds = np.array(self._cv.bounds.to_list()).reshape(2, -1).T
        self._bounds = cv_bounds.copy()
        self._bounds -= cv_bounds[:, 0:1]
        return self._bounds

    @property
    def chunk_id_bounds(self):
        return np.ceil((self.bounds / self._chunk_size[:, None])).astype(np.int)

    @property
    def layer_chunk_bounds(self):
        if self._layer_bounds_d:
            return self._layer_bounds_d
        layer_bounds_d = {}
        for layer in range(2, self.n_layers):
            layer_bounds = self.chunk_id_bounds / (2 ** (layer - 2))
            layer_bounds_d[layer] = np.ceil(layer_bounds).astype(np.int)
        self._layer_bounds_d = layer_bounds_d
        return self._layer_bounds_d

    @property
    def chunk_coord_gen(self):
        return itertools.product(*[range(*r) for r in self.chunk_id_bounds])

    @property
    def chunk_coords(self):
        if not self._chunk_coords is None:
            return self._chunk_coords
        self._chunk_coords = np.array(list(self.chunk_coord_gen), dtype=np.int)
        return self._chunk_coords

    @property
    def n_layers(self):
        if self._n_layers is None:
            self._n_layers = self.cg.n_layers
        return self._n_layers

    @property
    def use_raw_edge_data(self):
        return self._use_raw_edge_data

    @property
    def use_raw_agglomeration_data(self):
        return self._use_raw_agglomeration_data

    @property
    def edges_dir(self):
        return self._edges_dir

    @property
    def components_dir(self):
        return self._components_dir

    @property
    def task_q(self):
        if self._task_q:
            return self._task_q
        self._task_q = get_rq_queue(self._task_q_name)
        return self._task_q

    @property
    def build_graph(self):
        return self._build_graph

    def get_serialized_info(self, pickled=False):
        info = {
            "storage_path": self.storage_path,
            "cg_table_id": self._cg_table_id,
            "n_layers": self.n_layers,
            "instance_id": self._instance_id,
            "project_id": self._project_id,
            "data_version": self.data_version,
            "use_raw_edge_data": self._use_raw_edge_data,
            "use_raw_agglomeration_data": self._use_raw_agglomeration_data,
            "edges_dir": self._edges_dir,
            "components_dir": self._components_dir,
            "task_q_name": self._task_q_name,
            "build_graph": self._build_graph,
        }
        if pickled:
            return pickle.dumps(info)
        return info

    def is_out_of_bounds(self, chunk_coordinate):
        if not self._bitmasks:
            self._bitmasks = compute_bitmasks(self.n_layers, 2)
        return np.any(chunk_coordinate < 0) or np.any(
            chunk_coordinate > 2 ** self._bitmasks[1]
        )

    @classmethod
    def from_pickle(cls, serialized_info):
        return cls(**pickle.loads(serialized_info))

