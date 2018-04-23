# ChunkedGraph

## Credentials

The ChunkedGraph is essentially hosted by Google (BigTable). Currently, the credentials to access the data of 
any ChunkedGraph need to be stored at:
```
"~/.cloudvolume/secrets/google-secret.json"
```

## Usage

Initialization:
```
import chunkedgraph
cg = chunkedgraph.ChunkedGraph(non_dev=True)
```

### Acquiring the agglomeration id for an atomic supervoxel
```
agg_id = cg.read_agglomeration_id_with_atomic_id(atomic_id, time_stamp=time_stamp)
```
To get the most recent data `time_stamp` should be set to `None`. Otherwise, use the `time` package to create an `int` 
representing the time of the desired checkpoint. 

This function could also be used to acquire the agglomeration id 
for a supervoxel on other hierarchy levels.

