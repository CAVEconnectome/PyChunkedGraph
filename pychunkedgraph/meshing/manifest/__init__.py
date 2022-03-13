import numpy as np


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
