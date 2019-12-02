import numpy as np


def pad_encode_uint64(key: np.uint64):
    """
    Max uint64 has 20 digits.
    This functions pads zeros and encodes string to bytes.
    """
    return f"{key:020d}".encode("utf-8")
