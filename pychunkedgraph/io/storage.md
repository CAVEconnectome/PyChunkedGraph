## Serialization

PyChunkedgraph uses protobuf for serialization and zstandard for compression.

Edges are stored using the following protobuf definition.
You can also find it [here](https://github.com/seung-lab/PyChunkedGraph/blob/akhilesh-jobs-layer-dependency/pychunkedgraph/io/protobuf/chunkEdges.proto)

```
syntax = "proto3";

package edges;

message EdgesMsg {
  bytes node_ids1 = 1;
  bytes node_ids2 = 2;
  bytes affinities = 3;
  bytes areas = 4;
}

message ChunkEdgesMsg {
  EdgesMsg in_chunk = 1;
  EdgesMsg cross_chunk = 2;
  EdgesMsg between_chunk = 3;
}
```

This format is a result of multiple performance tests.
It provided the best tradeoff between deserialzation speed and storage size.

To read and write edges in this format, the functions `get_chunk_edges` and `put_chunk_edges`
in the module `pychunkedgraph.io.edges` may be used.

[CloudVolume](https://github.com/seung-lab/cloud-volume) is used for uploading and downloading this data. 

### Example usage

```
from pychunkedgraph.backend.edges import Edges

in_chunk = [[1,2],[2,3],[0,2],[2,4]]
between_chunk = [[1,5]]
cross_chunk = [[3,4]]

in_chunk_edges = Edges(in_chunk[:,0], in_chunk[:,1])
between_chunk_edges = Edges(between_chunk[:,0], between_chunk[:,1])
cross_chunk_edges = Edges(cross_chunk[:,0], cross_chunk[:,1])

edges_path = "<path_to_bucket>"
chunk_coordinates = np.array([0,0,0])

edges_d = {
  "in": in_chunk_edges,
  "between": between_chunk_edges,
  "cross": cross_chunk_edges
}

put_chunk_edges(edges_path, chunk_coordinates, edges_d, compression_level=22)
# file will be located at <path_to_bucket>/edges_0_0_0.proto.zst

# reading the file will simply return the previous dictionary
edges_d = get_chunk_edges(edges_path, [chunk_coordinates])

# notice the difference between chunk_coordinates parameter
# put_chunk_edges takes in coordinates for a single chunk
# get_chunk_edges takes in a list of chunk coordinates
```
