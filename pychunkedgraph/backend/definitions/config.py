from collections import namedtuple

datasource_fields = (
    "agglomeration",
    "watershed",
    "edges",
    "components",
    "use_raw_edges",
    "use_raw_components",
    "size",
)
DataSource = namedtuple(
    "DataSource", datasource_fields, defaults=(None,) * len(datasource_fields)
)

graphconfig_fields = ("graph_id", "chunk_size", "fanout")
GraphConfig = namedtuple(
    "GraphConfig", graphconfig_fields, defaults=(None,) * len(graphconfig_fields)
)

bigtableconfig_fields = ("project_id", "instance_id")
BigTableConfig = namedtuple(
    "BigTableConfig",
    bigtableconfig_fields,
    defaults=(None,) * len(bigtableconfig_fields),
)
