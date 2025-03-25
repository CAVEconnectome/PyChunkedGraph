import itertools
import numpy as np
import glob
import os
import time

import cloudvolume

from pychunkedgraph.creator import creator_utils
from multiwrapper import multiprocessing_utils as mu


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
            print("%d / %d - dt: %.3fs - eta: %.3fs" %
                  (i_fp, n_file_paths, dt, eta))

        rewrite_single_segmentation_block(fp, from_cv=from_cv, to_cv=to_cv)


def rewrite_single_segmentation_block(file_path, from_cv=None, to_cv=None,
                                      from_url=None, to_url=None):
    if from_cv is None:
        assert from_url is not None
        from_cv = cloudvolume.CloudVolume(from_url)

    if to_cv is None:
        assert to_url is not None
        assert 'svenmd' in to_url
        to_cv = cloudvolume.CloudVolume(to_url, bounded=False)

    dx, dy, dz, _ = os.path.basename(file_path).split("_")

    x_start, x_end = np.array(dx.split("-"), dtype=int)
    y_start, y_end = np.array(dy.split("-"), dtype=int)
    z_start, z_end = np.array(dz.split("-"), dtype=int)

    bbox = to_cv.bounds.to_list()[3:]
    if x_end > bbox[0]:
        x_end = bbox[0]

    if y_end > bbox[1]:
        y_end = bbox[1]

    if z_end > bbox[2]:
        z_end = bbox[2]

    seg = from_cv[x_start: x_end, y_start: y_end, z_start: z_end]
    mapping = creator_utils.read_mapping_h5(file_path)

    if 0 in seg and not 0 in mapping[:, 0]:
        mapping = np.concatenate(([np.array([[0, 0]], dtype=np.uint64), mapping]))

    sort_idx = np.argsort(mapping[:, 0])
    idx = np.searchsorted(mapping[:, 0], seg, sorter=sort_idx)
    out = np.asarray(mapping[:, 1])[sort_idx][idx]

    # print(out.shape, x_start, x_end, y_start, y_end, z_start, z_end)
    to_cv[x_start: x_end, y_start: y_end, z_start: z_end] = out


def rewrite_segmentation(dataset_name, n_threads=64, n_units_per_thread=None):
    if dataset_name == "pinky":
        cv_url = "gs://nkem/pinky40_v11/mst_trimmed_sem_remap/region_graph/"
        from_url = "gs://neuroglancer/pinky40_v11/watershed/"
        to_url = "gs://neuroglancer/svenmd/pinky40_v11/watershed/"
    elif dataset_name == "basil":
        cv_url = "gs://nkem/basil_4k_oldnet/region_graph/"
        from_url = "gs://neuroglancer/ranl/basil_4k_oldnet/ws/"
        to_url = "gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed/"
    else:
        raise Exception("Dataset unknown")

    file_paths = np.sort(glob.glob(creator_utils.dir_from_layer_name(
        creator_utils.layer_name_from_cv_url(cv_url)) + "/*rg2cg*"))

    if n_units_per_thread is None:
        file_path_blocks = np.array_split(file_paths, n_threads*3)
    else:
        n_blocks = int(np.ceil(len(file_paths) / n_units_per_thread))
        file_path_blocks = np.array_split(file_paths, n_blocks)

    multi_args = []
    for fp_block in file_path_blocks:
        multi_args.append([fp_block, from_url, to_url])

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(_rewrite_segmentation_thread, multi_args,
                             n_threads=n_threads, verbose=True,
                             debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_rewrite_segmentation_thread, multi_args,
                                n_threads=n_threads)


def _rewrite_image_thread(args):
    start_coordinates, end_coordinates, block_size, from_url, to_url, mip = args

    from_cv = cloudvolume.CloudVolume(from_url, mip=mip)
    to_cv = cloudvolume.CloudVolume(to_url, bounded=False, mip=mip)

    assert 'svenmd' in to_url

    coordinate_iter = itertools.product(np.arange(start_coordinates[0], end_coordinates[0], block_size[0]),
                                        np.arange(start_coordinates[1], end_coordinates[1], block_size[1]),
                                        np.arange(start_coordinates[2], end_coordinates[2], block_size[2]))

    for coordinate in coordinate_iter:
        rewrite_single_image_block(coordinate, block_size, from_cv=from_cv,
                                   to_cv=to_cv)


def rewrite_single_image_block(coordinate, block_size, from_cv=None, to_cv=None,
                               from_url=None, to_url=None, mip=None):
    if from_cv is None:
        assert from_url is not None and mip is not None
        from_cv = cloudvolume.CloudVolume(from_url, mip=mip)

    if to_cv is None:
        assert to_url is not None and mip is not None
        assert 'svenmd' in to_url
        to_cv = cloudvolume.CloudVolume(to_url, bounded=False, mip=mip,
                                        compress=False)

    x_start = coordinate[0]
    x_end = coordinate[0] + block_size[0]
    y_start = coordinate[1]
    y_end = coordinate[1] + block_size[1]
    z_start = coordinate[2]
    z_end = coordinate[2] + block_size[2]

    bbox = to_cv.bounds.to_list()[3:]
    if x_end > bbox[0]:
        x_end = bbox[0]

    if y_end > bbox[1]:
        y_end = bbox[1]

    if z_end > bbox[2]:
        z_end = bbox[2]

    print(x_start, y_start, z_start, x_end, y_end, z_end)

    img = from_cv[x_start: x_end, y_start: y_end, z_start: z_end]
    to_cv[x_start: x_end, y_start: y_end, z_start: z_end] = img


def rechunk_dataset(dataset_name, block_size=(1024, 1024, 64), n_threads=64,
                    mip=0):
    if dataset_name == "pinky40em":
        from_url = "gs://neuroglancer/pinky40_v11/image_rechunked/"
        to_url = "gs://neuroglancer/svenmd/pinky40_v11/image_512_512_32/"
    elif dataset_name == "pinky100seg":
        from_url = "gs://neuroglancer/nkem/pinky100_v0/ws/lost_no-random/bbox1_0/"
        to_url = "gs://neuroglancer/svenmd/pinky100_v0/ws/lost_no-random/bbox1_0_64_64_16/"
    elif dataset_name == "basil":
        raise()
    else:
        raise Exception("Dataset unknown")

    from_cv = cloudvolume.CloudVolume(from_url, mip=mip)

    dataset_bounds = np.array(from_cv.bounds.to_list())
    block_size = np.array(list(block_size))

    super_block_size = block_size * 2

    coordinate_iter = itertools.product(np.arange(dataset_bounds[0],
                                                  dataset_bounds[3],
                                                  super_block_size[0]),
                                        np.arange(dataset_bounds[1],
                                                  dataset_bounds[4],
                                                  super_block_size[1]),
                                        np.arange(dataset_bounds[2],
                                                  dataset_bounds[5],
                                                  super_block_size[2]))
    coordinates = np.array(list(coordinate_iter))

    multi_args = []
    for coordinate in coordinates:
        end_coordinate = coordinate + super_block_size
        m = end_coordinate > dataset_bounds[3:]
        end_coordinate[m] = dataset_bounds[3:][m]

        multi_args.append([coordinate, end_coordinate, block_size,
                           from_url, to_url, mip])

    # Run parallelizing
    if n_threads == 1:
        mu.multiprocess_func(_rewrite_image_thread, multi_args,
                             n_threads=n_threads, verbose=True,
                             debug=n_threads == 1)
    else:
        mu.multisubprocess_func(_rewrite_image_thread, multi_args,
                                n_threads=n_threads)