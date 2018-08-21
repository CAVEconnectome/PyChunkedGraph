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

    if 'unbreakable' in path:
        dt = 'uint64, uint64'
    elif 'isolated' in path:
        dt = 'uint64'
    else:
        dt = 'uint64, uint64, float32, uint64'

    buf = cv_st.get_file(path)
    edge_data = np.frombuffer(buf, dtype=dt)

    if len(edge_data) == 0:
        if len(dt.split(",")) == 1:
            edge_data = np.array([], dtype=np.uint64)
        else:
            edge_data = {"f0": np.array([], dtype=np.uint64),
                         "f1": np.array([], dtype=np.uint64),
                         "f2": np.array([], dtype=np.float32),
                         "f3": np.array([], dtype=np.uint64)}

    if 'isolated' in path:
        edge_dict = {"node_ids": edge_data}
    else:
        edge_ids = np.concatenate([edge_data["f0"].reshape(-1, 1),
                                   edge_data["f1"].reshape(-1, 1)], axis=1)

        edge_dict = {"edge_ids": edge_ids}

    if "connected" in path:
        edge_dict['edge_affs'] = edge_data['f2']
        edge_dict['edge_areas'] = edge_data['f3']

    return edge_dict


def read_edge_file_h5(path, layer_name=None):
    if not path.endswith(".h5"):
        path = path[:-4] + ".h5"

    if layer_name is not None:
        path = dir_from_layer_name(layer_name) + path

    edge_dict = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            edge_dict[k] = f[k].value

    return edge_dict


def download_and_store_edge_file(cv_st, path, create_dir=True):
    edge_dict = read_edge_file_cv(cv_st, path)

    dir_path = dir_from_layer_name(layer_name_from_cv_url(cv_st.layer_path))

    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    with h5py.File(dir_path + path[:-4] + ".h5", "w") as f:
        for k in edge_dict.keys():
            f.create_dataset(k, data=edge_dict[k], compression="gzip")


# def read_mapping_cv(cv_st, path, olduint32=False):
#     """ Reads the mapping information from a file """
#
#     if olduint32:
#         mapping = np.frombuffer(cv_st.get_file(path),
#                                 dtype=np.uint64).reshape(-1, 2)
#         mapping_to = mapping[:, 1]
#         mapping_from = np.frombuffer(np.ascontiguousarray(mapping[:, 0]), dtype=np.uint32)[::2].astype(np.uint64)
#         return np.concatenate([mapping_from[:, None], mapping_to[:, None]], axis=1)
#     else:
#         return np.frombuffer(cv_st.get_file(path), dtype=np.uint64).reshape(-1, 2)
#
#
# def read_mapping_h5(path, layer_name=None):
#     if not path.endswith(".h5"):
#         path = path[:-4] + ".h5"
#
#     if layer_name is not None:
#         path = dir_from_layer_name(layer_name) + path
#
#     with h5py.File(path, "r") as f:
#         mapping = f["mapping"].value
#
#     return mapping
#
#
# def download_and_store_mapping_file(cv_st, path, olduint32=False):
#     mapping = read_mapping_cv(cv_st, path, olduint32=olduint32)
#
#     dir_path = dir_from_layer_name(layer_name_from_cv_url(cv_st.layer_path))
#
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#
#     with h5py.File(dir_path + path[:-4] + ".h5", "w") as f:
#         f["mapping"] = mapping