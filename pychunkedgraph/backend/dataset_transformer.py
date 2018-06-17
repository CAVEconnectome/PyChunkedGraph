import numpy as np
import glob
import os
import time

import cloudvolume

from . import utils
from . import multiprocessing_utils as mu


def _rewrite_segmentation_thread(args):
    file_paths, from_url, to_url = args

    from_cv = cloudvolume.CloudVolume(from_url)
    to_cv = cloudvolume.CloudVolume(to_url, bounded=False)

    assert 'svenmd' in to_url

    n_file_paths = len(file_paths)

    time_start = time.time()
    for i_fp, fp in enumerate(file_paths):
        if i_fp % 10 == 5:
            dt = time.time() - time_start
            eta = dt / i_fp * n_file_paths - dt
            print("%d / %d - dt: %.3fs - eta: %.3fs" % (i_fp, n_file_paths, dt, eta))

        rewrite_single_block(fp, from_cv=from_cv, to_cv=to_cv)


def rewrite_single_block(file_path, from_cv=None, to_cv=None, from_url=None,
                         to_url=None):
    if from_cv is None:
        assert from_url is not None
        from_cv = cloudvolume.CloudVolume(from_url)

    if to_cv is None:
        assert to_url is not None
        assert 'svenmd' in to_url
        to_cv = cloudvolume.CloudVolume(to_url, bounded=False)

    dx, dy, dz, _ = os.path.basename(file_path).split("_")

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
    mapping = utils.read_mapping_h5(file_path)

    if 0 in seg and not 0 in mapping[:, 0]:
        mapping = np.concatenate(([np.array([[0, 0]], dtype=np.uint64), mapping]))

    sort_idx = np.argsort(mapping[:, 0])
    idx = np.searchsorted(mapping[:, 0], seg, sorter=sort_idx)
    out = np.asarray(mapping[:, 1])[sort_idx][idx]

    # print(out.shape, x_start, x_end, y_start, y_end, z_start, z_end)
    to_cv[x_start: x_end, y_start: y_end, z_start: z_end] = out


def rewrite_segmentation(cv_url="gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/",
                         from_url="gs://neuroglancer/pinky40_v11/watershed/",
                         to_url="gs://neuroglancer/svenmd/pinky40_v11/watershed/",
                         n_threads=64, n_units_per_thread=None):

    file_paths = np.sort(glob.glob(utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*rg2cg*"))

    if n_units_per_thread is None:
        file_path_blocks = np.array_split(file_paths, n_threads*3)
    else:
        n_blocks = int(np.ceil(len(file_paths) / n_units_per_thread))
        file_path_blocks = np.array_split(file_paths, n_blocks)

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
