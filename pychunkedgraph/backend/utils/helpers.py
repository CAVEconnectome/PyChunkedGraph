from typing import Sequence, Tuple

import numpy as np

def get_bounding_box(
    source_coords: Sequence[Sequence[int]],
    sink_coords: Sequence[Sequence[int]],
    bb_offset: Tuple[int, int, int] = (120, 120, 12),
):
    bb_offset = np.array(list(bb_offset))
    source_coords = np.array(source_coords)
    sink_coords = np.array(sink_coords)

    coords = np.concatenate([source_coords, sink_coords])
    bounding_box = [np.min(coords, axis=0), np.max(coords, axis=0)]
    bounding_box[0] -= bb_offset
    bounding_box[1] += bb_offset
    return bounding_box    
