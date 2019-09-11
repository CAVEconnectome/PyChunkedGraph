"""
Agglomeration
"""

from typing import Optional

import numpy as np

from .edges import Edges


class Agglomeration:
    """
    An agglomeration is a connected component at a given level.
    Composed of supervoxel ids and the edges between them.
    """

    def __init__(self, supervoxels: np.ndarray, edges: Edges, level: Optional[int] = 2):
        self.supervoxels = supervoxels
        self.edges = edges
        self.level = level
