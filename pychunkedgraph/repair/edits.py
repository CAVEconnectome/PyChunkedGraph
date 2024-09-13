# pylint: disable=protected-access,missing-function-docstring,invalid-name,wrong-import-position

from datetime import timedelta

from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.attributes import Concurrency
from pychunkedgraph.graph.operation import GraphEditOperation


def _get_previous_log_ts(cg, operation):
    log, previous_ts = cg.client.read_log_entry(operation - 1)
    if log:
        return previous_ts
    return _get_previous_log_ts(cg, operation - 1)


def repair_operation(
    cg: ChunkedGraph,
    operation_id: int,
    unlock: bool = False,
    use_preceding_edit_ts=True,
) -> GraphEditOperation.Result:
    operation = GraphEditOperation.from_operation_id(
        cg, operation_id, multicut_as_split=False, privileged_mode=True
    )

    _, current_ts = cg.client.read_log_entry(operation_id)
    parent_ts = current_ts - timedelta(milliseconds=10)
    if operation_id > 1 and use_preceding_edit_ts:
        previous_ts = _get_previous_log_ts(cg, operation_id)
        parent_ts = previous_ts + timedelta(milliseconds=100)

    result = operation.execute(
        operation_id=operation_id,
        parent_ts=parent_ts,
        override_ts=current_ts + timedelta(milliseconds=1),
    )
    old_roots = operation._update_root_ids()

    if unlock:
        for root_ in old_roots:
            cg.client.unlock_root(root_, result.operation_id)
            cg.client.unlock_indefinitely_locked_root(root_, result.operation_id)
    return result


if __name__ == "__main__":
    op_ids_to_retry = [...]
    locked_roots = [...]

    cg = ChunkedGraph(graph_id="<graph_id>")
    node_attrs = cg.client.read_nodes(node_ids=locked_roots)
    for node_id, attrs in node_attrs.items():
        if Concurrency.IndefiniteLock in attrs:
            locked_op = attrs[Concurrency.IndefiniteLock][0].value
            op_ids_to_retry.append(locked_op)
            print(f"{node_id} indefinitely locked by op {locked_op}")
    print(f"total to retry: {len(op_ids_to_retry)}")

    logs = cg.client.read_log_entries(op_ids_to_retry)
    for op_id, log in logs.items():
        print(f"repairing {op_id}")
        repair_operation(cg, log, op_id)
