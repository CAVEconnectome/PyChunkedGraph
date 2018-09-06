# Multiprocessing

`multiprocessing_utils.py` contains functionality to execute functions (that do not need to communicate with 
each other) in parallel, i.e. the same functions gets called with different parameter sets.

The function `func` needs to accept an array of arguments only. Example from `chunkcreator.py`:

```
def _add_layer_thread(args):
    dev_mode, layer_id, chunk_coords = args

    cg = chunkedgraph.ChunkedGraph(dev_mode=dev_mode)
    cg.add_layer(layer_id, chunk_coords)
```
Hence `args` needs to be a list of lists - number of jobs x number of arguments. 

There are two ways to start processes in `multiprocessing_utils.py`. The standard approach use Python's 
`multiprocessing.pool` to start multiple processes, the other creates `subprocesses` via the command line. In general,
the former approach should be chosen if there are no significant reasons to choose 'subprocesses' (such as 
library limitations). 

The functions can be called with

```
multiprocessing_utils.multiprocess_func(func, args, nb_cpus=nb_cpus, debug=nb_cpus==1)

```

and 

```
multiprocessing_utils.multisubprocess_func(func, args, nb_cpus=nb_cpus)
```

`multisubprocess_func` creates a whole folder hierarchy, copies the repository and runs individual Python calls 
for each process.

The current implementation waits until all jobs have finished (using `pool.close()` and `pool.join()` or `p.wait()`).
If you do not require the jobs to be finished when continuing with the main program you could remove these in a forked 
version.
