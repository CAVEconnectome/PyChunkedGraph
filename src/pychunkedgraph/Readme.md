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

## Usage

### Initialization
```
import chunkedgraph
cg = chunkedgraph.ChunkedGraph(non_dev=True)
```

Currently, this initializes a ChunkedGraph corresponding to (1k, 1k, 1k) voxel volume. I will develop the library in
in the `non_dev=False` version of the ChunkedGraph and try to keep the `non_dev=True` version on the same 
developmental level as the version on github.

### Acquiring the agglomeration id for an atomic supervoxel
```
agg_id = cg.read_agglomeration_id_with_atomic_id(atomic_id, time_stamp=time_stamp, is_cg_id=False)
```
To get the most recent data `time_stamp` should be set to `None`. Otherwise, use the `time` package to create an `int` 
representing the time of the desired checkpoint. There is a discrepancy between the ids used in neuroglancer which
correspond to the ones in the region graph and the ids used in the ChunkedGraph. The ChunkedGraph is able to map these
ids onto each other. The user has to specificy whether the given id is from the region graph (`is_cg_id=False`) or
from the ChunkedGraph (`is_cg_id=True`). Mapping the region graph id to the chunked graph id (`is_cg_id=False`) incurs
an additional read from the database.

`read_agglomeration_id_with_atomic_id` can also be used to acquire the agglomeration id  for a supervoxel on other hierarchy levels.


### Historic agglomeration ids

The ChunkedGraph stores the history for any agglomeration id to identify whether different agglomeration ids 
correspond to different versions of the same agglomerated object. 

```
historical_agglomeration_ids = cg.read_agglomeration_id_history(agglomeration_id, time_stamp=time_stamp)
```

The user can limit the time window in which historical agglomeration ids should be retrived 
(`time_stamp=earliest_time`). 

While the ChunkedGraph does not support writes, this simply returns the given agglomeration id.
