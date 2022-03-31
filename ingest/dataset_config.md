Configuration for ingest must be specified in a YaML file such as this:

```
data_source:
  EDGES: ""
  COMPONENTS: ""
  WATERSHED: ""

graph_config:
  CHUNK_SIZE: [X, Y, Z]
  FANOUT: <int>
  SPATIAL_BITS: <int>
  LAYER_ID_BITS: <int>

backend_client:
  TYPE: "bigtable"
  CONFIG:
    ADMIN: true
    READ_ONLY: false
```

### `data_source`
* `EDGES` - path to edge files.
* `COMPONENTS` - path to component files.
* `WATERSHED` - path to flat segmentation for which the chunkedgraph is being created. Must have an `info` file in the path that specifies volume size among other things.

The protocol for these paths must be supported by [cloud-files](https://github.com/seung-lab/cloud-files/) or [cloud-volume](https://github.com/seung-lab/cloud-volume/).

### `graph_config`
* `CHUNK_SIZE` - atomic chunk size in [x, y, z].
* `FANOUT` - number of chunks in each axis that form a larger parent chunk.
* `SPATIAL_BITS` - number of bits in segment IDs reserved per axis for chunk coordinates.
* `LAYER_ID_BITS` - number of bits in segment IDs reserved for layer.

For more information refer to this well documented [graphene](https://github.com/seung-lab/cloud-volume/wiki/Graphene) section in the cloud-volume wiki, courtesy of Will Silversmith.

### `backend_client`
This can be left as is. Currently bigtable is the only supported backend to store the chunkedgraph.

