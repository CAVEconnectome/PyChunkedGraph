import datetime
import numpy as np

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, NamedTuple

from multiwrapper import multiprocessing_utils as mu

from pychunkedgraph.backend.chunkedgraph_utils import get_google_compatible_time_stamp
from pychunkedgraph.backend.utils import column_keys
from pychunkedgraph.backend import chunkedgraph

def add_edges(cg, operation_id: np.uint64,
              atomic_edges: Sequence[Sequence[np.uint64]],
              time_stamp: datetime.datetime,
              affinities: Optional[Sequence[np.float32]] = None
              ):

    atomic_edges = np.array(atomic_edges, dtype=np.uint64)

    # Comply to resolution of BigTables TimeRange
    time_stamp = get_google_compatible_time_stamp(time_stamp,
                                                  round_up=False)

    if affinities is None:
        affinities = np.ones(len(atomic_edges),
                             dtype=column_keys.Connectivity.Affinity.basetype)

    assert len(affinities) == len(atomic_edges)

    rows = []


def propagate_edits_to_root(cg: chunkedgraph.ChunkedGraph,
                            lvl2_dict: Dict,
                            operation_id: np.uint64,
                            time_stamp: datetime.datetime):
    """

    :param cg: ChunkedGraph
    :param lvl2_dict: dict
        maps
    :param operation_id:
    :param time_stamp:
    :return:
    """

    eh = EditHelper(cg, lvl2_dict)




class EditHelper(object):
    def __init__(self, cg, lvl2_dict):
        self._cg = cg
        self._lvl2_dict = lvl2_dict

        self._parent_dict = {}
        self._children_dict = {}
        self._cross_chunk_edge_dict = {}

    @property
    def cg(self):
        return self._cg

    @property
    def lvl2_dict(self):
        return self._lvl2_dict

    def get_children(self, node_id):
        if not node_id in self._children_dict:
            self._children_dict[node_id] = self.get_children(node_id)
            for child_id in self._children_dict[node_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = node_id
                else:
                    assert self._parent_dict[child_id] == node_id

        return self._children_dict[node_id]

    def get_parent(self, node_id):
        if not node_id in self._parent_dict:
            self._parent_dict[node_id] = self.get_parent(node_id)

        return self._children_dict[node_id]

    def read_cross_chunk_edges(self, node_id):
        if not node_id in self._cross_chunk_edge_dict:
            self._cross_chunk_edge_dict[node_id] = \
                self.cg.read_cross_chunk_edges(node_id)

        return self._cross_chunk_edge_dict[node_id]

    def bulk_family_read(self):
        def _get_root_thread(lvl2_node_id):
            parent_ids = self.cg.get_root(lvl2_node_id, get_all_parents=True)
            parent_ids = np.concatenate([[lvl2_node_id], parent_ids])

            for i_parent in range(len(parent_ids) - 1):
                self._parent_dict[parent_ids[i_parent]] = parent_ids[i_parent+1]

        lvl2_node_ids = []
        for v in self.lvl2_dict.values():
            lvl2_node_ids.extend(v)

        mu.multithread_func(_get_root_thread, lvl2_node_ids,
                            n_threads=len(lvl2_node_ids), debug=False)

        parent_ids = list(self._parent_dict.keys())
        child_dict = self.cg.get_children(parent_ids, flatten=False)

        for parent_id in child_dict:
            self._children_dict[parent_id] = child_dict[parent_id]

            for child_id in self._children_dict[parent_id]:
                if not child_id in self._parent_dict:
                    self._parent_dict[child_id] = parent_id
                else:
                    assert self._parent_dict[child_id] == parent_id

    def bulk_cross_chunk_edge_read(self):
        raise NotImplementedError


