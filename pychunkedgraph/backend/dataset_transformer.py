import numpy as np
import glob
import os

import cloudvolume

from . import utils
from . import multiprocessing_utils as mu


def _rewrite_segmentation_thread(args):
    file_paths, from_url, to_url = args

    from_cv = cloudvolume.CloudVolume(from_url)
    to_cv = cloudvolume.CloudVolume(to_url, bounded=False)

    for fp in file_paths:
        print(fp)
        dx, dy, dz, _ = os.path.basename(fp).split("_")

        x_start, x_end = np.array(dx.split("-"), dtype=np.int)
        y_start, y_end = np.array(dy.split("-"), dtype=np.int)
        z_start, z_end = np.array(dz.split("-"), dtype=np.int)

        bbox = to_cv.bounds.to_list()[3:]
        if x_end > bbox[0]:
            x_end = bbox[0]

        if y_end > bbox[1]:
            y_end = bbox[1]

        if z_end > bbox[2]:
            z_end = bbox[2]

        seg = from_cv[x_start: x_end, y_start: y_end, z_start: z_end]
        mapping = utils.read_mapping_h5(fp)

        sort_idx = np.argsort(mapping[:, 0])
        idx = np.searchsorted(mapping[:, 0], seg, sorter=sort_idx)
        out = np.asarray(mapping[:, 1])[sort_idx][idx]

        print(out.shape, x_end, y_end, z_end)

        to_cv[x_start: x_end, y_start: y_end, z_start: z_end] = out


def rewrite_segmentation(cv_url="gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/",
                         from_url="gs://neuroglancer/pinky40_v11/watershed/",
                         to_url="gs://neuroglancer/svenmd/pinky40_v11/watershed/",
                         n_threads=64):

    file_paths = np.sort(glob.glob(utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*rg2cg*"))

    file_path_blocks = np.array_split(file_paths, n_threads*3)

    multi_args = []
    for fp_block in file_path_blocks:
        multi_args.append([fp_block, from_url, to_url])

    # Run multiprocessing
    if n_threads == 1:
        mu.multiprocess_func(_rewrite_segmentation_thread, multi_args,
                             n_threads=n_threads, verbose=True,
                             debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_rewrite_segmentation_thread, multi_args,
                                n_threads=n_threads)
