import numpy as np
from cloudvolume import Storage

from pychunkedgraph.backend.chunkedgraph import ChunkedGraph  # noqa



def get_downstream_multi_child_node(cg: ChunkedGraph, node_id: np.uint64,
                                    stop_layer: int = 1):
    """
    Return the first descendant of `node_id` (including itself) with more than
    one child, or the first descendant of `node_id` (including itself) on or
    below layer `stop_layer`.
    """
    layer = cg.get_chunk_layer(node_id)
    if layer <= stop_layer:
        return node_id

    children = cg.get_children(node_id)
    if len(children) > 1:
        return node_id

    if not children:
        raise ValueError(f"Node {node_id} on layer {layer} has no children.")

    return get_downstream_multi_child_node(cg, children[0], stop_layer)


def get_highest_child_nodes_with_meshes(cg: ChunkedGraph, node_id: np.uint64,
                                        stop_layer=1):
    test_ids = [get_downstream_multi_child_node(cg, node_id, stop_layer)]
    valid_seg_ids = []

    with Storage("%s/%s" % (cg.cv.layer_cloudpath, cg.cv.info["mesh"])) as stor:
        while len(test_ids) > 0:
            file_paths = ["%d:0" % seg_id for seg_id in test_ids]
            test_ids = []

            existence_dict = stor.files_exist(file_paths)

            for k in existence_dict:
                seg_id = np.uint64(int(k[:-2]))
                if existence_dict[k]:
                    valid_seg_ids.append(seg_id)
                else:
                    if cg.get_chunk_layer(seg_id) > stop_layer:
                        test_ids.extend(cg.get_children(seg_id))

    return valid_seg_ids