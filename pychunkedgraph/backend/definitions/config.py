from collections import namedtuple

_datasource_fields = (
    "agglomeration",
    "watershed",
    "edges",
    "components",
    "use_raw_edges",
    "use_raw_components",
    "data_version",
)
_datasource_defaults = (None, None, None, None, True, True, 2)
DataSource = namedtuple("DataSource", _datasource_fields, defaults=_datasource_defaults)

_graphconfig_fields = (
    "graph_id",
    "chunk_size",
    "fanout",
    "build_graph",
    "s_bits_atomic_layer",
)
_graphconfig_defaults = (None, None, 2, True, 8)
GraphConfig = namedtuple(
    "GraphConfig", _graphconfig_fields, defaults=_graphconfig_defaults
)

_bigtableconfig_fields = ("project_id", "instance_id")
_bigtableconfig_defaults = ("neuromancer-seung-import", "pychunkedgraph")
BigTableConfig = namedtuple(
    "BigTableConfig", _bigtableconfig_fields, defaults=_bigtableconfig_defaults
)
