import dill
import numpy as np


def apply_log(cg, cl_path):
    with open(cl_path, "rb") as f:
        cl = dill.load(f)

    for cl_key in np.sort(list(cl.keys())):
        cl_entry = cl[cl_key]

        if cl_entry["is_split"]:
            ret = cg.remove_edges(user_id="default",
                                  source_ids=None,
                                  sink_ids=None,
                                  source_coords=cl_entry["source_coords"],
                                  sink_coords=cl_entry["sink_coords"],
                                  mincut=True)
        else:
            ret = cg.add_edges(user_id="default",
                               atomic_edges=np.array([cl_entry["source_coords"][0],
                                                      cl_entry["sink_coords"][0]],
                                                     dtype=np.uint64),
                               source_coord=cl_entry["source_coords"],
                               sink_coord=cl_entry["sink_coords"],
                               return_new_lvl2_nodes=True,
                               remesh_preview=False)




