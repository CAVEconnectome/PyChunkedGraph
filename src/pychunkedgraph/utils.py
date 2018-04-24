import numpy as np


def read_edge_file_cv(cv_st, path):
    """ Reads the edge ids and affinities from an edge file """

    dt = np.dtype('uint64')
    # dt = dt.newbyteorder('>')
    edge_buffer = np.frombuffer(cv_st.get_file(path), dtype=dt).reshape(-1, 3)
    edge_ids = edge_buffer[:, :2]
    edge_affs = np.frombuffer(edge_buffer[:, -1].tobytes(), dtype=np.float32)[::2]

    return edge_ids, edge_affs


def read_mapping(cv_st, path):
    """ Reads the mapping information from a file """

    return np.frombuffer(cv_st.get_file(path), dtype=np.uint64).reshape(-1, 2)


