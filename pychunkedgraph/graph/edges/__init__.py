"""
Classes and types for edges
"""

from .definitions import EDGE_TYPES, Edges
from .ocdbt import put_edges, get_edges

from .stale import (
    get_new_nodes,
    get_stale_nodes,
    get_latest_edges,
    get_latest_edges_wrapper,
)
