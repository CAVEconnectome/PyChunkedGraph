# pylint: disable=invalid-name, missing-docstring, unused-wildcard-import, wildcard-import
#
# Backward-compatible re-export facade.
# All functionality has been split into:
#   meshgen_utils.py    — shared Draco, segmentation, merge utilities and constants
#   meshgen_initial.py  — initial ingest mesh pipeline
#   meshgen_remesh.py   — dynamic remeshing pipeline

from pychunkedgraph.meshing.meshgen_utils import (  # noqa: F401
    UTC,
    PRINT_FOR_DEBUGGING,
    WRITING_TO_CLOUD,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_PASSWORD,
    REDIS_URL,
    decode_draco_mesh_buffer,
    remap_seg_using_unsafe_dict,
    calculate_stop_layer,
    get_meshing_necessities_from_graph,
    calculate_quantization_bits_and_range,
    get_draco_encoding_settings_for_chunk,
    get_next_layer_draco_encoding_settings,
    transform_draco_vertices,
    transform_draco_fragment_and_return_encoding_options,
    merge_draco_meshes_across_boundaries,
    black_out_dust_from_segmentation,
    get_multi_child_nodes,
)

from pychunkedgraph.meshing.meshgen_initial import (  # noqa: F401
    get_remapped_segmentation,
    get_higher_to_lower_remapping,
    get_root_lx_remapping,
    get_lx_overlapping_remappings,
    chunk_initial_mesh_task,
    chunk_initial_sharded_stitching_task,
)

from pychunkedgraph.meshing.meshgen_remesh import (  # noqa: F401
    get_remapped_seg_for_lvl2_nodes,
    get_root_remapping_for_nodes_and_svs,
    get_lx_overlapping_remappings_for_nodes_and_svs,
    remeshing,
    chunk_stitch_remeshing_task,
)
