# pylint: disable=invalid-name, missing-docstring, not-callable

import os

import numpy as np
import pyfqmr


def _get_simplification_parameters(lod, n_faces):
    assert lod >= 2, "2 is the highest lod."

    l2factor = int(os.environ.get("l2factor", "2"))
    factor = int(os.environ.get("factor", "4"))
    aggressiveness = float(os.environ.get("aggr", "7.0"))

    if lod == 2:
        target_count = max(int(n_faces / l2factor), 4)
    else:
        target_count = max(int(n_faces / factor), 4)

    return aggressiveness, target_count


def simplify(cg, fragment, fragment_id, lod=None):
    """
    Simplify with pyfqmr; input and output flat vertices and faces.
    """
    vertices = fragment["vertices"].reshape(-1, 3)
    faces = fragment["faces"].reshape(-1, 3)

    layer = cg.get_chunk_layer(fragment_id)
    lod = layer if lod is None else lod
    aggressiveness, target_count = _get_simplification_parameters(lod, len(faces))

    scale = (2 ** (layer - 2)) * np.array(cg.meta.graph_config.CHUNK_SIZE, dtype=int)
    vertices = vertices / scale

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(
        target_count=target_count,
        aggressiveness=aggressiveness,
        preserve_border=True,
        verbose=False,
    )
    vertices, faces, _ = simplifier.getMesh()
    vertices = vertices * scale

    fragment["num_vertices"] = len(vertices)
    fragment["vertices"] = vertices.flatten()
    fragment["faces"] = faces.flatten()
    return fragment


def simplify_skipped_layers(cg, fragment, fragment_id, fragment_parent_id):
    """
    A node can have children at any layer below it due to skip connections.
    This can lead to parts of mesh having a higher level of detail.
    To make the mesh uniform, such child fragment will need to be simplified repeatedly.
    For eg: consider a node at l6 with one l5 and one l2 children.
    l2 fragment will have a higher level of detail because it is not simplified.
    l5 has been simplified thrice (l3, l4 and l5).
    The l2 fragment will need to be simplified to achieve a similar level of detail.
    """

    parent_layer = cg.get_chunk_layer(fragment_parent_id)
    layer = cg.get_chunk_layer(fragment_id)

    layer += 1
    while layer < parent_layer:
        fragment = simplify(cg, fragment, fragment_id, lod=layer)
        layer += 1
    fragment["vertices"] = fragment["vertices"].reshape(-1, 3)
    fragment["faces"] = fragment["faces"].reshape(-1, 3)
    return fragment
