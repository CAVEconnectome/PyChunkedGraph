# ChunkedGraph

## Installation

You need `google-cloud` to use the ChunkedGraph

```
pip install google-cloud
```

as well as `cloud-volume`

```
pip install cloud-volume
```


## Credentials

The ChunkedGraph is essentially hosted by Google (BigTable). Currently, the credentials to access the data of 
any ChunkedGraph need to be stored at:
```
"~/.cloudvolume/secrets/google-secret.json"
```

## Data

Currently, **only** supervoxels contained in `gs://nkem/pinky40_agglomeration_test_1024_2/region_graph` are included in the
ChunkedGraph. The chunk size is `[512, 512, 64]`. 

**WARNING**: There are many false edges in the chunkedgraph due to a bug in the ingested data. This leads to an overly merged. As soon as the problem is fixed by Nico, the ChunkedGraph will be updated to the new data and will cover all of pinky40.


## Usage

### Initialization
```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=False)
```

Currently, this initializes a ChunkedGraph corresponding to (1k, 1k, 1k) voxel volume. I will develop the library in
in the `dev_mode=True` version of the ChunkedGraph and try to keep the `dev_mode=False` (default option) version on the same 
developmental level as the version on github.

### Acquiring the root id for an atomic supervoxel id
```
root_id = cg.get_root(atomic_id, time_stamp=time_stamp, is_cg_id=False)
```
To get the most recent data `time_stamp` should be set to `None` (default option). Otherwise, use the `time` package to create an `int`  representing the time of the desired checkpoint. There is a discrepancy between the ids used in neuroglancer which correspond to the ones in the region graph and the ids used in the ChunkedGraph. The ChunkedGraph is able to map these ids onto each other. 

The user has to specificy whether the given id is from the region graph (`is_cg_id=False`, default option) or from the ChunkedGraph (`is_cg_id=True`). Mapping the region graph id to the chunked graph id (`is_cg_id=False`) incurs
an additional read from the database.

`get_root` can also be used to acquire the root id  for a supervoxel on other hierarchy levels.

#### Example

```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(non_dev=True)

atomic_id = 1028242092029
root_id = cg.get_root(atomic_id, time_stamp=time_stamp, is_cg_id=False)
```

### Historic agglomeration ids

The ChunkedGraph stores the history for any agglomeration id to identify whether different agglomeration ids 
correspond to different versions of the same agglomerated object. 

```
historical_agglomeration_ids = cg.read_agglomeration_id_history(agglomeration_id, time_stamp=time_stamp)
```

The user can limit the time window in which historical agglomeration ids should be retrived 
(`time_stamp=earliest_time`). 

While the ChunkedGraph does not support writes, this simply returns the given agglomeration id.
