from typing import Iterable, Dict
from google.cloud import bigtable

import numpy as np


def pad_node_id(node_id: np.uint64) -> str:
    """ Pad node id to 20 digits

    :param node_id: int
    :return: str
    """
    return "%.20d" % node_id


def serialize_uint64(node_id: np.uint64) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    return serialize_key(pad_node_id(node_id))  # type: ignore


def serialize_uint64s_to_regex(node_ids: Iterable[np.uint64]) -> bytes:
    """ Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    node_id_str = "".join(["%s|" % pad_node_id(node_id)
                           for node_id in node_ids])[:-1]
    return serialize_key(node_id_str)  # type: ignore


def deserialize_uint64(node_id: bytes) -> np.uint64:
    """ De-serializes a node id from a BigTable row

    :param node_id: bytes
    :return: np.uint64
    """
    return np.uint64(node_id.decode())  # type: ignore


def serialize_key(key: str) -> bytes:
    """ Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: bytes
    """
    return key.encode("utf-8")


def deserialize_key(key: bytes) -> str:
    """ Deserializes a row key

    :param key: bytes
    :return: str
    """
    return key.decode()


def row_to_byte_dict(row: bigtable.row.Row, f_id: str = None, idx: int = None
                     ) -> Dict[int, Dict]:
    """ Reads row entries to a dictionary

    :param row: row
    :param f_id: str
    :param idx: int
    :return: dict
    """
    row_dict = {}

    for fam_id in row.cells.keys():
        row_dict[fam_id] = {}

        for row_k in row.cells[fam_id].keys():
            if idx is None:
                row_dict[fam_id][deserialize_key(row_k)] = \
                    [c.value for c in row.cells[fam_id][row_k]]
            else:
                row_dict[fam_id][deserialize_key(row_k)] = \
                    row.cells[fam_id][row_k][idx].value

    if f_id is not None and f_id in row_dict:
        return row_dict[f_id]
    elif f_id is None:
        return row_dict
    else:
        raise Exception("Family id not found")