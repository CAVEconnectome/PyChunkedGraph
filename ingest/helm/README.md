### Ingesting a ChunkedGraph
After the segmentation process is complete, agglomeration data is used to create a chunkedgraph to create a proof-readable dataset.

### Prerequisites
#### Tools
* GCloud SDK
* Kubernetes client (kubectl)
* [Helm](https://helm.sh/docs/intro/install/)
#### Infrastructure:
* Bigtable instance^
* Bucket to store atomic edges and componenents^
* Kubernetes cluster#
* Managed redis instance# (1-2G depending on dataset size)

^only need to be created once per project.

\# must be on the same network.

:memo: All infrastructure must be in the same region for best performance.

:memo: Make sure to increase Bigtable nodes to handle increased read/writes during the process. For small datasets 30 seem to work well, for bigger datasets (eg: V1DD) a count of 40+ seems necessary.

The cluster is recommended to have 2 pools - `master` and `workers`. The names are important because kubernetes uses them to schedule pods for master and worker deployments. `master` pools should be on-demand, workers can be pre-emptible.

Both atomic and higher layer jobs can use the same worker pool (details below). For datasets such as `vnc` nodes with 16G memory is enough but for larger datasets like `v1dd` the memory requirement increases as you go higher up the layer (as high as 200-300G). The cluster needs to be reconfigured as ingest progresses. Because of this reason, ingest is best done layer by layer, starting with atomic layer (layer 2).

:memo: The number of worker nodes should not be too high, recommended max is 250. Bigtable struggles to distribute the load evenly when there are too many workers (hot node performance degrades).

### Ingest

A service account key (JSON) with appropriate permissions is required. Create `/ingest/chart/secrets/google-secret.json` and paste the contents of the key in the JSON file (named `google-secret.json` because cloud-volume looks for it).

Next a yaml file with ingest configuration is required. For eg, `vnc1_full_v3align_2.yml`:
```
data_source:
  EDGES: "gs://chunkedgraph/vnc1_full_v3align_2/realigned_v1/0-86016_0-225780_10-4400/edges"
  COMPONENTS: "gs://chunkedgraph/vnc1_full_v3align_2/realigned_v1/0-86016_0-225780_10-4400/components"
  WATERSHED: "gs://zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/ws/full_run_v1"
  DATA_VERSION: 4

graph_config:
  CHUNK_SIZE: [512, 512, 128]
  FANOUT: 2
  SPATIAL_BITS: 10
  ID_PREFIX: ""

backend_client:
  TYPE: "bigtable"
  CONFIG:
    ADMIN: true
    READ_ONLY: false
    PROJECT: "zetta-lee-fly-vnc-001"
    INSTANCE: "chunkedgraphs"

ingest_config:
  AGGLOMERATION: >-
    gs://seuronbot_zetta_lee_fly_vnc_001/vnc1_full_v3align_2/realigned_v1/0-86016_0-225780_10-4400/full_run_v1/agg
```

Place this in `/ingest/chart/config/vnc1_full_v3align_2.yml`, and:
```
$ cd ingest/chart/
$ helm install ingest . --debug
```
This will install the chart (takes a few seconds). Then `exec` into the master pod and queue jobs as follows:
```
$ kubectl exec -ti deploy/master -- bash
container> flask ingest graph <name> config/vnc1_full_v3align_2.yml --raw
```
This will take a while for small datasets. For larger datasets, the process goes to sleep for `60s` after queuing `100K` jobs to avoid blowing up redis memory, repeats until all layer 2 jobs are queued.

While this continues, atomic worker deployment can be started with:
```
$ helm upgrade ingest . --debug --set atomicWorkers.enabled=true --set atomicWorkers.hpa.minReplicas=250
```
Then in another shell login to master pod to check on progress:
```
$ kubectl exec -ti deploy/master -- bash
container> flask rq status atomic (to track job status)
NOTE: Use --show-busy to display count of non idle workers
Queue name 	: atomic
Jobs queued 	: xxxxx
Workers total 	: 0
Jobs failed 	: 0
container> flask ingest status (to track jobs progress on each layer)
...
```
`watch` can be used with the above two `flask` commands for continuous monitoring. This takes care of atomic layer (layer 2).

#### Layer 3+
After layer 2 has successfully completed, upgrade the helm installation to stop atomic workers and start parent layer (abstract) workers.
```
$ helm upgrade ingest . --debug --set atomicWorkers.enabled=false --set atomicWorkers.hpa.minReplicas=1
$ helm upgrade ingest . --debug --set abstractWorkers.enabled=true --set abstractWorkers.hpa.minReplicas=250
```
Then, again in the `master` container shell run the following command:
```
container> flask ingest layer 3
```
Use the same command for higher layers after each previous layer jobs finish successfully.



