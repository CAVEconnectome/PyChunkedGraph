# pylint: disable=invalid-name, missing-docstring

import numpy as np

from .cache import ManifestCache
from .sharded import get_children_before_start_layer
from .sharded import verified_manifest as verified_manifest_sharded
from .sharded import speculative_manifest as speculative_manifest_sharded


def get_highest_child_nodes_with_meshes(
    cg,
    node_id: np.uint64,
    start_layer: int,
    bounding_box=None,
):
    return verified_manifest_sharded(
        cg,
        node_id,
        start_layer=start_layer,
        bounding_box=bounding_box,
    )
