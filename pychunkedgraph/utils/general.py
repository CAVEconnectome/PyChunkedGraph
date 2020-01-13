"""
generic helper funtions
"""
from typing import Sequence


import numpy as np


def reverse_dictionary(dictionary):
    """
    given a dictionary - {key1 : [item1, item2 ...], key2 : [ite3, item4 ...]}
    return {item1: key1, item2: key1, item3: key2, item4: key2 }
    """
    keys = []
    vals = []
    for key, values in dictionary.items():
        keys.append([key] * len(values))
        vals.append(values)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)

    return {k: v for k, v in zip(vals, keys)}


def chunked(l: Sequence, n: int):
    """Yield successive n-sized chunks from l."""
    if n < 1:
        n = len(l)
    for i in range(0, len(l), n):
        yield l[i : i + n]


def in2d(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    arr1_view = arr1.view(dtype="u8,u8").reshape(arr1.shape[0])
    arr2_view = arr2.view(dtype="u8,u8").reshape(arr2.shape[0])
    return np.in1d(arr1_view, arr2_view)
