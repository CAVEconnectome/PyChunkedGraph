from collections import namedtuple

DataSource = namedtuple(
    "DataSource", ["agglomeration", "watershed", "edges", "components", "size"]
)

GraphConfig = namedtuple(
    "GraphConfig", ["graph_id", "chunk_size", "fanout"]
)

BigTableConfig = namedtuple("BigTableConfig", ["project_id", "instance_id"])
