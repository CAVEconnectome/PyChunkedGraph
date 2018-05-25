import numpy as np
import glob
import os

import cloudvolume

from . import utils


def rewrite_segmentation(cv_url="gs://nkem/basil_4k_oldnet/region_graph/",
                         from_url="gs://neuroglancer/ranl/basil_4k_oldnet/ws",
                         to_url="gs://neuroglancer/svenmd/basil_4k_oldnet_cg/watershed"):

    from_cv = cloudvolume.CloudVolume(from_url)
    to_cv = cloudvolume.CloudVolume(to_url, bounded=False)

    file_paths = np.sort(glob.glob(utils.dir_from_layer_name(utils.layer_name_from_cv_url(cv_url)) + "/*rg2cg*"))

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


