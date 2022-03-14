from time import time

import numpy as np
from cloudvolume import Storage

from .sharded import speculative_manifest as speculative_manifest_sharded
from ..meshgen_utils import get_mesh_name


def get_highest_child_nodes_with_meshes(
    cg,
    node_id: np.uint64,
    start_layer: int,
    stop_layer: int = 2,
    bounding_box=None,
    verify_existence=False,
    flexible_start_layer=None,
):
    from .sharded import verified_manifest as verified_manifest_sharded

    # TODO support for unsharded mesh based on dataset meta (low priority)
    return verified_manifest_sharded(
        cg,
        node_id,
        start_layer=start_layer,
        bounding_box=bounding_box,
    )


def children_meshes_non_sharded(
    cg,
    node_id: np.uint64,
    stop_layer=2,
    start_layer=None,
    verify_existence=False,
    bounding_box=None,
    flexible_start_layer=None,
):
    if flexible_start_layer is not None:
        # Get highest children that are at flexible_start_layer or below
        # (do this because of skip connections)
        candidates = cg.get_children_at_layer(node_id, flexible_start_layer, True)
    elif start_layer is None:
        candidates = np.array([node_id], dtype=np.uint64)
    else:
        candidates = cg.get_subgraph(
            node_id,
            bbox=bounding_box,
            bbox_is_coordinate=True,
            return_layers=[start_layer],
            nodes_only=True,
        )

    if verify_existence:
        valid_node_ids = []
        with Storage(cg.cv_mesh_path) as stor:  # pylint: disable=not-context-manager
            while True:
                filenames = [get_mesh_name(cg, c) for c in candidates]

                start = time()
                existence_dict = stor.files_exist(filenames)
                print("Existence took: %.3fs" % (time() - start))

                missing_meshes = []
                for mesh_key in existence_dict:
                    node_id = np.uint64(mesh_key.split(":")[0])
                    if existence_dict[mesh_key]:
                        valid_node_ids.append(node_id)
                    else:
                        if cg.get_chunk_layer(node_id) > stop_layer:
                            missing_meshes.append(node_id)

                start = time()
                if missing_meshes:
                    candidates = cg.get_children(missing_meshes, flatten=True)
                else:
                    break
                print("ChunkedGraph lookup took: %.3fs" % (time() - start))
    else:
        valid_node_ids = candidates
    return valid_node_ids, [get_mesh_name(cg, s) for s in valid_node_ids]
