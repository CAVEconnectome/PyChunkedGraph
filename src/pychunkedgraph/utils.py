import numpy as np
import h5py
import os

HOME = os.path.expanduser("~")


def layer_name_from_cv_url(cv_url):
    return cv_url.strip("/").split("/")[-2]


def dir_from_layer_name(layer_name):
    return HOME + "/" + layer_name + "/"


def read_edge_file_cv(cv_st, path):
    """ Reads the edge ids and affinities from an edge file """

    dt = np.dtype('uint64')
    # dt = dt.newbyteorder('>')
    edge_buffer = np.frombuffer(cv_st.get_file(path), dtype=dt).reshape(-1, 3)
    edge_ids = edge_buffer[:, :2]
    edge_affs = np.frombuffer(edge_buffer[:, -1].tobytes(), dtype=np.float32)[::2]

    return edge_ids, edge_affs


def read_edge_file_h5(path, layer_name=None):
    if not path.endswith(".h5"):
        path = path[:-4] + ".h5"

    if layer_name is not None:
        path = dir_from_layer_name(layer_name) + path

    with h5py.File(path, "r") as f:
        edge_ids = f["edge_ids"].value
        edge_affs = f["edge_affs"].value

    return edge_ids, edge_affs


def download_and_store_edge_file(cv_st, path, create_dir=True):
    edge_ids, edge_affs = read_edge_file_cv(cv_st, path)

    dir_path = dir_from_layer_name(layer_name_from_cv_url(cv_st.layer_path))

    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    with h5py.File(dir_path + path[:-4] + ".h5", "w") as f:
        f["edge_ids"] = edge_ids
        f["edge_affs"] = edge_affs


def read_mapping_cv(cv_st, path):
    """ Reads the mapping information from a file """

    return np.frombuffer(cv_st.get_file(path), dtype=np.uint64).reshape(-1, 2)


def read_mapping_h5(path, layer_name=None):
    if not path.endswith(".h5"):
        path = path[:-4] + ".h5"

    if layer_name is not None:
        path = dir_from_layer_name(layer_name) + path

    with h5py.File(path, "r") as f:
        mapping = f["mapping"].value

    return mapping


def download_and_store_mapping_file(cv_st, path):
    mapping = read_mapping_cv(cv_st, path)

    dir_path = dir_from_layer_name(layer_name_from_cv_url(cv_st.layer_path))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with h5py.File(dir_path + path[:-4] + ".h5", "w") as f:
        f["mapping"] = mapping