from cloudvolume import CloudVolume, Storage
import numpy as np
import os
from pychunkedgraph.meshing import meshgen, meshgen_utils


def compute_centroid_by_range(vertices):
    bbox_min = np.amin(vertices, axis=0)
    bbox_max = np.amax(vertices, axis=0)
    return bbox_min + ((bbox_max - bbox_min) / 2)


def compute_centroid_with_chunk_boundary(cg, vertices, l2_id, last_l2_id):
    """
    Given a level 2 id, the vertices of its mesh, and the level 2 id preceding it in
    a path, return the center point of the mesh on the chunk boundary separating the two
    ids, and the center point of the entire mesh.
    :param cg: ChunkedGraph object
    :param vertices: [[np.float]]
    :param l2_id: np.uint64
    :param last_l2_id: np.uint64 or None
    :return: [np.float]
    """
    centroid_by_range = compute_centroid_by_range(vertices)
    if last_l2_id is None:
        return [centroid_by_range]
    l2_id_cc = cg.get_chunk_coordinates(l2_id)
    last_l2_id_cc = cg.get_chunk_coordinates(last_l2_id)

    # Given the coordinates of the two level 2 ids, find the chunk boundary
    axis_change = 2
    look_for_max = True
    if l2_id_cc[0] != last_l2_id_cc[0]:
        axis_change = 0
    elif l2_id_cc[1] != last_l2_id_cc[1]:
        axis_change = 1
    if np.sum(l2_id_cc - last_l2_id_cc) > 0:
        look_for_max = False
    if look_for_max:
        value_to_filter = np.amax(vertices[:, axis_change])
    else:
        value_to_filter = np.amin(vertices[:, axis_change])
    chunk_boundary_vertices = vertices[
        np.where(vertices[:, axis_change] == value_to_filter)
    ]

    # Get the center point of the mesh on the chunk boundary
    bbox_min = np.amin(chunk_boundary_vertices, axis=0)
    bbox_max = np.amax(chunk_boundary_vertices, axis=0)
    return [bbox_min + ((bbox_max - bbox_min) / 2), centroid_by_range]


def compute_mesh_centroids_of_l2_ids(cg, l2_ids, flatten=False):
    """
    Given a list of l2_ids, return a tuple containing a dict that maps l2_ids to their
    mesh's centroid (a global coordinate), and a list of the l2_ids for which the mesh does not exist.
    :param cg: ChunkedGraph object
    :param l2_ids: Sequence[np.uint64]
    :return: Union[Dict[np.uint64, np.ndarray], [np.uint64], [np.uint64]]
    """
    cv_sharded_mesh_dir = cg.meta.dataset_info["mesh"]
    cv_unsharded_mesh_dir = cg.meta.dataset_info["mesh_metadata"]["unsharded_mesh_dir"]
    cv_unsharded_mesh_path = os.path.join(
        cg.meta.data_source.WATERSHED, cv_sharded_mesh_dir, cv_unsharded_mesh_dir
    )
    cv = CloudVolume(
        f"graphene://https://localhost/segmentation/table/dummy",
        mesh_dir=cv_sharded_mesh_dir,
        info=meshgen_utils.get_json_info(cg),
    )
    fragments_to_fetch = [
        f"{l2_id}:0:{meshgen_utils.get_chunk_bbox_str(cg, cg.get_chunk_id(l2_id))}"
        for l2_id in l2_ids
    ]
    if flatten:
        centroids_with_chunk_boundary_points = []
    else:
        centroids_with_chunk_boundary_points = {}
    last_l2_id = None
    failed_l2_ids = []
    with Storage(cv_unsharded_mesh_path) as storage:
        files_contents = storage.get_files(fragments_to_fetch)
        fragment_map = meshgen.get_missing_initial_meshes(cv, files_contents)
        for i in range(len(fragments_to_fetch)):
            fragment_to_fetch = fragments_to_fetch[i]
            l2_id = l2_ids[i]
            try:
                fragment = fragment_map[fragment_to_fetch]
                if fragment["content"] is not None and fragment["error"] is None:
                    if "skip_decode" in fragment:
                        vertices = fragment["content"].vertices
                    else:
                        mesh = meshgen.decode_draco_mesh_buffer(fragment["content"])
                        vertices = mesh["vertices"]
                    if flatten:
                        centroids_with_chunk_boundary_points.extend(
                            compute_centroid_with_chunk_boundary(
                                cg, vertices, l2_id, last_l2_id
                            )
                        )
                    else:
                        centroids_with_chunk_boundary_points[
                            l2_id
                        ] = compute_centroid_with_chunk_boundary(
                            cg, vertices, l2_id, last_l2_id
                        )
            except:
                failed_l2_ids.append(l2_id)
            last_l2_id = l2_id
    return centroids_with_chunk_boundary_points, failed_l2_ids