

from pychunkedgraph.exporting import export


def _log_type(log_entry):
    if "removed_edges" in log_entry:
        return "split"
    else:
        return "merge"

def apply_log(cg, log):
    assert cg.table_id != "pinky100_sv11"

    last_operation_id = -1
    for operation_id in log.keys():
        assert last_operation_id < int(operation_id)

        log_entry = log[operation_id]

        print(log_entry)

        if _log_type(log_entry) == "merge":
            print("MERGE")
            if len(log_entry["added_edges"]) == 0:
                affinities = None
            else:
                affinities = log_entry["added_edges"]

            cg.add_edges(user_id=log_entry["user"],
                         atomic_edges=log_entry["added_edges"],
                         affinities=affinities,
                         source_coord=log_entry["source_coords"],
                         sink_coord=log_entry["sink_coords"],
                         n_tries=60)
        elif _log_type(log_entry) == "split":
            print("SPLIT")
            cg.remove_edges(user_id=log_entry["user"],
                            source_ids=log_entry["source_ids"],
                            sink_ids=log_entry["sink_ids"],
                            source_coords=log_entry["source_coords"],
                            sink_coords=log_entry["sink_coords"],
                            atomic_edges=log_entry["removed_edges"],
                            mincut=False,
                            bb_offset=log_entry["bb_offset"],
                            n_tries=20)
        else:
            raise NotImplementedError

        last_operation_id = int(operation_id)