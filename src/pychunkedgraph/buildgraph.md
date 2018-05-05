# Creating a ChunkedGraph

There are two steps to creating a ChunkedGraph for a region graph:
0. Creating the table and BigTable family
1. Downloading files from `cloudvolume` and storing them on disk
2. Creating the ChunkedGraph from these files

## Creating the table and family

Deleting the current table:

```
from src.pychunkedgraph import chunkedgraph
cg = chunkedgraph.ChunkedGraph(dev_mode=True)

cg.table.delete()
```

Creating a new table and family:

```
cg = chunkedgraph.ChunkedGraph(dev_mode=True)

cg.table.create()
f = cg.table.column_family(cg.family_id) 
f.create()
```

## Downloading all files from cloudvolume

To download all relevant friles from a cloudvolume directory do

```
from src.pychunkedgraph import chunkcreator

chunkcreator.download_and_store_cv_files(cv_url)
```
The files are stored as h5's in a directory in `home`. The directory name is chosen to be the layer name.


## Building the ChunkedGraph

```
chunkcreator.create_chunked_graph(cv_url, dev_mode=False, nb_cpus=1)
```

`dev_mode` is an easy way to switch between two different tables, one meant for production and one for development. `nb_cpus` 
allows the user to run this process in parallel using `subprocesses` (see [multiprocessing.md](https://github.com/seung-lab/PyChunkedGraph/blob/master/src/pychunkedgraph/multiprocessing.md)).




