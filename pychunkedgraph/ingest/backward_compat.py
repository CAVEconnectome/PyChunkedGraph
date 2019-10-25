from typing import Dict

import numpy as np

from .ran_agglomeration import define_active_edges
from ..backend.edges import EDGE_TYPES


def get_chunk_data(chunk_edges_all: Dict, mapping: Dict):
    active_edges_flag_d, isolated_ids = define_active_edges(chunk_edges_all, mapping)

    edge_ids = {}
    edge_affs = {}
    edge_areas = {}
    for k in EDGE_TYPES:
        edges = chunk_edges_all[k]
        if k == "cross":
            edge_ids[k] = np.concatenate(
                [edges.node_ids1[:, None], edges.node_ids2[:, None]], axis=1
            )
            continue

        sv1_conn = edges.node_ids1[active_edges_flag_d[k]]
        sv2_conn = edges.node_ids2[active_edges_flag_d[k]]
        aff_conn = edges.affinities[active_edges_flag_d[k]]
        area_conn = edges.areas[active_edges_flag_d[k]]
        edge_ids[f"{k}_connected"] = np.concatenate(
            [sv1_conn[:, None], sv2_conn[:, None]], axis=1
        )
        edge_affs[f"{k}_connected"] = aff_conn.astype(np.float32)
        edge_areas[f"{k}_connected"] = area_conn

        sv1_disconn = edges.node_ids1[~active_edges_flag_d[k]]
        sv2_disconn = edges.node_ids2[~active_edges_flag_d[k]]
        aff_disconn = edges.affinities[~active_edges_flag_d[k]]
        area_disconn = edges.areas[~active_edges_flag_d[k]]
        edge_ids[f"{k}_disconnected"] = np.concatenate(
            [sv1_disconn[:, None], sv2_disconn[:, None]], axis=1
        )
        edge_affs[f"{k}_disconnected"] = aff_disconn.astype(np.float32)
        edge_areas[f"{k}_disconnected"] = area_disconn

    return edge_ids, edge_affs, edge_areas, isolated_ids
