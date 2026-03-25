"""Shared dataclasses for the proposed stitch algorithm."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StitchResult:
    new_roots: list
    new_l2_ids: list
    new_ids_per_layer: dict
    entries: list
    perf: dict = field(default_factory=dict)
    ctx: Any = None


@dataclass
class StitchContext:
    """Accumulated state passed through all stitch phases."""
    l2_edges: np.ndarray
    l2ids: np.ndarray
    l2_cx_edges: dict
    children_d: dict
    atomic_cx: dict
    resolver: dict
    sv_to_l2: dict
    old_hierarchy: dict
    reader: Any = None
    new_ids_d: dict = field(default_factory=lambda: defaultdict(list))
    old_to_new: dict = field(default_factory=dict)
    node_cx: dict = field(default_factory=dict)
    l2_atomic_cx: dict = field(default_factory=dict)
    sibling_ids: set = field(default_factory=set)
    new_node_ids: set = field(default_factory=set)
    children_cache: dict = field(default_factory=dict)
    parents_cache: dict = field(default_factory=dict)
    cx_cache: dict = field(default_factory=dict)
    siblings_d: dict = field(default_factory=lambda: defaultdict(list))


@dataclass
class RunResult:
    """Result of a single-file stitch run. Separates JSON-serializable metadata from structure."""
    structure: dict
    new_roots: list
    elapsed: float
    graph_id: str
    n_edges: int
    layer_counts: dict
    perf: dict = field(default_factory=dict)
    new_l2_ids: list = field(default_factory=list)
    new_ids_per_layer: dict = field(default_factory=dict)
    n_entries_written: int = 0
    operation_id: Any = None
    table_name: str = ""

    @property
    def meta(self) -> dict:
        """JSON-serializable metadata (excludes structure)."""
        return {k: v for k, v in self.__dict__.items() if k != "structure"}
