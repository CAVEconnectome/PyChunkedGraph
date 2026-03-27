from dataclasses import dataclass, field


@dataclass
class StitchResult:
    new_roots: list
    new_l2_ids: list
    new_ids_per_layer: dict
    rows: dict
    perf: dict = field(default_factory=dict)


@dataclass
class RunResult:
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
    table_name: str = ""

    @property
    def meta(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "structure"}
