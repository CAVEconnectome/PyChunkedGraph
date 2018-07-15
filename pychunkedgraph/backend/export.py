import numpy as np

from . import chunkedgraph
from . import multiprocessing_utils as mu
from . import utils


def get_sv_to_root_id_mapping_chunk(cg, chunk_coords):
    """ Acquires a svid -> rootid dictionary for a chunk

    :param cg: chunkedgraph instance
    :param chunk_coords: list
    :return: dict
    """
    sv_to_root_mapping = {}

    chunk_coords = np.array(chunk_coords, dtype=np.int)

    if np.any((chunk_coords % cg.chunk_size) != 0):
        raise Exception("Chunk coords have to match a chunk corner exactly")

    chunk_coords = chunk_coords / cg.chunk_size
    chunk_coords = chunk_coords.astype(np.int)
    bb = np.array([chunk_coords, chunk_coords + 1], dtype=np.int)

    print(bb)

    atomic_rows = cg.range_read_chunk(layer=1, x=chunk_coords[0],
                                      y=chunk_coords[1], z=chunk_coords[2])It
    for atomic_key in atomic_rows.keys():
        atomic_id = chunkedgraph.deserialize_node_id(atomic_key)

        # Check if already found the root for this supervoxel
        if atomic_id in sv_to_root_mapping:
            continue

        # Find root
        root_id = cg.get_root(atomic_id)
        sv_to_root_mapping[atomic_id] = root_id

        # Add atomic children of root_id
        atomic_ids = cg.get_subgraph(root_id, bounding_box=bb,
                                     bb_is_coordinate=False)
        sv_to_root_mapping.update(dict(zip(atomic_ids, [root_id] * len(atomic_ids))))


def write_flat_segmentation(cg, cv_path_chunkedgraph, cv_path_flat_segmentation,
                            bounding_box):
    """ Applies the mapping in the chunkedgraph to the supervoxels to create
        a flattened segmentation

    :param cg: chunkedgraph instance
    :param cv_path_chunkedgraph: str
    :param cv_path_flat_segmentation: str
    :param bounding_box: np.array
    :return: bool
    """

    pass

