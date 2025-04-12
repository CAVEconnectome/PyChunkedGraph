# ChunkedGraph

## Installation

The ChunkedGraph requires `google-cloud`, `cloud-volume` and `networkx`.

```
pip install -r requirements.txt
```

## Multiprocessing

Check out [multiprocessing.md](https://github.com/seung-lab/PyChunkedGraph/blob/master/src/pychunkedgraph/multiprocessing.md) for how to use the multiprocessing tools implemented for the ChunkedGraph


## Credentials

The ChunkedGraph is essentially hosted by Google (BigTable). Currently, the credentials to access the data of 
any ChunkedGraph need to be stored at:
```
"~/.cloudvolume/secrets/google-secret.json"
```

## Data

The current version of the ChunkedGraph contains supervoxels from `gs://nkem/basil_4k_oldnet/region_graph/`. The chunk size is `[512, 512, 64]`. 


## Usage

### Building the graph

[buildgraph.md](https://github.com/seung-lab/PyChunkedGraph/blob/master/src/pychunkedgraph/buildgraph.md) explains how to build a graph from scratch.


### Initialization
```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=False)
```

Currently, this initializes a ChunkedGraph corresponding to (1k, 1k, 1k) voxel volume. I will develop the library in
in the `dev_mode=True` version of the ChunkedGraph and try to keep the `dev_mode=False` (default option) version on the same 
developmental level as the version on github.


### ChunkedGraph versus region graph ids

The region graph is the original data representation defining edges between all supervoxels. The ids shown in neuroglancer are region graph ids. The ChunkedGraph reassignes ids to all supervoxels (internal representation). This is necessary because supervoxels are cut at chunk borders and supervoxels in the CHunkedGraph carry information about the chunk they are in. To transform the ids from the region graph to the ChunkedGraph when calling a function, define `is_cg_id=False`. The ChunkedGraph maintains entries for all region graph ids as lookups.


### Acquiring the root id for an atomic supervoxel id
```
root_id = cg.get_root(atomic_id, time_stamp=time_stamp, is_cg_id=False)
```
To get the most recent data `time_stamp` should be set to `None` (default option). Otherwise, use the `datetime` package to create a date representing the time of the desired checkpoint. There is a discrepancy between the ids used in neuroglancer which correspond to the ones in the region graph and the ids used in the ChunkedGraph. The ChunkedGraph is able to map these ids onto each other.

The user has to specificy whether the given id is from the region graph (`is_cg_id=False`, default option) or from the ChunkedGraph (`is_cg_id=True`). Mapping the region graph id to the chunked graph id (`is_cg_id=False`) incurs
an additional read from the database.

`get_root` can also be used to acquire the root id  for a supervoxel on other hierarchy levels.

##### Minimal example

```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=False)

atomic_id = 537753696
root_id = cg.get_root(atomic_id)
```

### Adding an atomic edge

```
cg.add_edge(edge_ids, is_cg_id=False)
```

To add an atomic edge, the user defines the two atomic segments that are connected. The ChunkedGraph will automatically set the `time_stamp` of the change to the current time point.


##### Minimal example

```
from src.pychunkedgraph import chunkedgraph
import numpy as np

cg = chunkedgraph.ChunkedGraph(dev_mode=False)

edge = np.array([537753696, 537544567], dtype=np.uint64)
cg.add_edge(edge)
```

### Removing edges

```
cg.remove_edge(edge_ids, is_cg_id=False)
```

`remove_edge` takes either a list of edges or a single edge as input. The ChunkedGraph will automatically set the `time_stamp` of the change to the current time point.

##### Minimal example

```
from src.pychunkedgraph import chunkedgraph
import numpy as np

cg = chunkedgraph.ChunkedGraph(dev_mode=False)

edge = np.array([2953055848, 2953047401], dtype=np.uint64)
cg.add_edge(edge)
cg.remove_edge(edge)
```

The easiest way to test this function is to add and then remove an edge.

### Historic agglomeration ids

The ChunkedGraph stores the history for any agglomeration id to identify whether different agglomeration ids 
correspond to different versions of the same agglomerated object. 

```
historical_agglomeration_ids = cg.read_agglomeration_id_history(agglomeration_id, time_stamp=time_stamp)
```

The user can limit the time window in which historical agglomeration ids should be retrived 
(`time_stamp=earliest_time`).

##### Minimal example
```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=False)

root_id = 432345564227584182
id_history = cg.read_agglomeration_id_history(root_id)
```


### Read the local ChunkedGraph

To read the edge list and edge affinities of all atomic super voxels belonging to a root node do
```
cg.get_subgraph(root_id, bounding_box, bbox_is_coordinate=True)
```

The user can define a `bounding_box=[[x_l, y_l, z_l], [x_h, y_h, z_h]]` as either coordinates or chunk id range (use `bbox_is_coordinate`). The `bounding_box` feature is currently not working, the parameter is ignored. The current datset is small enough to all reads of atomic supervoxels. Hence, this should not hinder any development.

##### Minimal example
```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=False)

root_id = 432345564227567621
edges, affinities = cg.get_subgraph(root_id)
```


### Time stamp

Several functions of the ChunkedGraph take a `time_stamp` as argument. The `time_stamp` is a [`datetime`](https://docs.python.org/2/library/datetime.html) object that can be created as follows:

```
import datetime

time_stamp = datetime.datetime.now()
```

`UTC` is set as timezone if none is defined.



