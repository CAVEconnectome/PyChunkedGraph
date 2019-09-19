from collections import namedtuple

DataSource = namedtuple(
    "DataSource", ["agglomeration", "watershed", "edges", "components"]
)

GraphConfig = namedtuple(
    "GraphConfig", ["graph_id", "chunk_size", "fan_out", "size", "is_new"]
)

BigTableConfig = namedtuple("BigTableConfig", ["project_id", "instance_id"])
