from collections import namedtuple

datasource_fields = (
    "agglomeration",
    "watershed",
    "edges",
    "components",
    "use_raw_edges",
    "use_raw_components",
    "data_version",
)
datasource_defaults = (None, None, None, None, True, True, 2)
DataSource = namedtuple("DataSource", datasource_fields, defaults=datasource_defaults)

graphconfig_fields = (
    "graph_id",
    "chunk_size",
    "fanout",
    "build_graph",
    "s_bits_atomic_layer",
)
graphconfig_defaults = (None, None, 2, True, 8)
GraphConfig = namedtuple(
    "GraphConfig", graphconfig_fields, defaults=graphconfig_defaults
)

bigtableconfig_fields = ("project_id", "instance_id")
BigTableConfig = namedtuple(
    "BigTableConfig",
    bigtableconfig_fields,
    defaults=(None,) * len(bigtableconfig_fields),
)
