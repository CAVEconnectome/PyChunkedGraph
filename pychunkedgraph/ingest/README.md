## Sample Ingest Run

### Credentials
To run ingest a service account key with appropriate permissions is required.

* BigTable - create table and write to tables.
* GCS or S3 - read and write objects if data is on public cloud.
* (Optional) MemoryStore Redis - read and write to managed redisdb instance.

### Environment Variables
```
BIGTABLE_PROJECT=<>
BIGTABLE_INSTANCE=<>
FLASK_APP=run_dev.py
AUTH_DISABLED=true
PCG_GRAPH_IDS=minnie3_v1
APP_SETTINGS=pychunkedgraph.app.config.DockerDevelopmentConfig
GOOGLE_APPLICATION_CREDENTIALS=/path/to/.cloudvolume/secrets/google-secret.json
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=dev
```

### Example Ingest Configuration

```
# filename: config.yml
data_source:
  # paths to edges and components from which to build chunkedgraph
  EDGES: "gs://chunked-graph/f7ed4db1fb26d94fb40fcebcd7975e3c/edges"
  COMPONENTS: "gs://chunked-graph/f7ed4db1fb26d94fb40fcebcd7975e3c/components"

  # path to watershed of the dataset
  WATERSHED: "gs://akhilesh-pcg/ranl/ws/f7ed4db1fb26d94fb40fcebcd7975e3c"

  # this is seunglab specific, can be ignored
  DATA_VERSION: 4

graph_config:
  # fundamental chunk size (layer 2)
  CHUNK_SIZE: [200, 200, 40]

  # number of chunks in each dimensio to form a bigger chunk
  # for eg: fanout=2 means 2^3 chunks will form a larger chunk at next layer
  FANOUT: 2

  # number of bits reserved to encode layer
  LAYER_ID_BITS: 8

  #number of bits used for each spatial id creation on level 1
  SPATIAL_BITS: 10

backend_client:
  TYPE: "bigtable"
  CONFIG:
    ADMIN: true
    READ_ONLY: false

ingest_config:
  AGGLOMERATION: >-
    gs://akhilesh-pcg/ranl/rgf/f7ed4db1fb26d94fb40fcebcd7975e3c/agg
```

Assume the above configuration is in a file named `config.yml`. Before we start ingest, redis server must be started. Redis is used as a job queue - [python-rq](https://python-rq.org/).

```
$ redis-server --requirepass <your_password>
```
Then on a different shell start the ingest process with this command:

```
$ flask ingest graph <unique__test_name> config.yml --test
```
`--test` flag will enqueue 8 chunks at the center of the dataset most likely to be not empty, so this is useful to do before running a full ingest.

Chunk jobs are enqueued starting at layer 2. Each chunk is one job. After running the ingest command above, use ingest status command to monitor progress.

```
$ watch flask ingest status
Every 2.0s: flask ingest status

2       : 0 / 847 # progress at each layer: completed_chunks / total_chunk_jobs
3       : 0 / 144
4       : 0 / 18
5       : 0 / 4
6       : 0 / 1
```

After the test tasks are enqueued, start a worker. 8 chunks should be enqueued because of the `--test` flag, this command would start an [rq worker](https://python-rq.org/docs/workers/) listening to the queue named `layer_2` and execute 8 jobs.
```
$ rq worker layer_2 layer_3 ... layer_N
```

To enqueue parent chunks automatically when its children chunks are completed, run a tracking worker which checks if children chunks are completed and their parent can be enqueued.

```
$ rq worker tracker
```
This will start a worker that listens to `tracker` queue. Each chunk job publishes to `tracker` queue when it completes so tracker worker can check if its parent can be enqueued. For instance, when all the 8 test chunks complete, the tracker worker will enqueue their parent on the queue `layer_3`.

If test chunks run successfully, repeat the same steps without the `--test` flag. Remember to flush redis db to delete old data.

```
$ flask ingest flush_redis
$ flask ingest graph <unique_name> config.yml
```
