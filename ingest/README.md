## Infrastructure
 `terraform` is used to create the infrastructure needed to run ingest. Currently scripts are provided to run the ingest easily on Google Cloud, but it can be run locally or on another cloud provider with appropriate setup.

### Requirements
* GCloud SDK (378.0.0)
* Terraform (v1.1.7)
* Helm (v3.7.0)

### Terraform ([docs](https://www.terraform.io/docs))

> IMPORTANT: This setup assumes that a bigtable instance is already created. To reduce latency, it is recommended that all resources are co-located in the same region as bigtable instance.

Provided scripts create a VPC network, subnet, redis instance, cluster with separately managed pools to run master and workers.

Customize variables in the file `terraform/terraform.tfvars` to create infrastructure in your Google Cloud project.

Run the `terraform apply` command to create resources.

```
$ cd terraform/
$ terraform apply
```
This will output some variables useful for next steps:
```
kubernetes_cluster_context = "gcloud container clusters get-credentials chunkedgraph-ingest --zone us-east1-b --project neuromancer-seung-import"
kubernetes_cluster_name = "chunkedgraph-ingest"
project_id = "neuromancer-seung-import"
redis_host = "10.128.211.211"
region = "us-east1"
zone = "us-east1-b"
```

Use value of `kubernetes_cluster_context` to connect to your cluster.
Use value of `redis_host` in `helm/values.yaml` (more info in Helm section).

### Helm ([docs](https://helm.sh/docs/))
`helm` is used to run the ingest. The provided chart installs kubernetes resources such as configmaps, secrets, deployments needed to run the ingest. Refer to example `helm/example_values.yaml` file for more information.

#### Helm Resources
When all variables are ready, rename your values file to `values.yaml` (ignored by git because it can contain sensitive information).
Then run:

```
$ cd helm/
$ helm install <release_name> . --debug --dry-run
```
If successful run the same command without `--dry-run`. This will create master and worker kubernetes deployments.
Pods will have dataset mounted in `/app/datasets` and `/app` is the `WORKDIR`.

Pods should now be in `Running` status, provided there were no issues. Run the following to create a bigtable and enqueue jobs.
```
$ kubectl exec -ti deploy/master -- bash
// now you're in the container
> ingest graph <unique_test_bigtable_name> datasets/test.yml --test
```

[RQ](https://python-rq.org/docs/) is used to create jobs. The package uses `redis` as a task queue.

The `--test` flag will queue 8 children chunks that share the same parent. When the children chunk jobs finish, worker listening on the `t2` (tracker for layer 2) queue should enqueue the parent chunk.

> NOTE: to avoid race conditions there should only be one worker listening on tracker queues for each layer. The provided helm chart makes sure of this but importnant not to forget.

You can check the progress of ingest with:
```
> ingest status
2       : 8 / 64 # progress at each layer
3       : 1 / 8
4       : 0 / 1
```
Output should look like this if succesful. Now you can rerun the ingest without the `--test` flag (make sure to use a different bigtable name).

Sometimes jobs fail for any number of reasons. Assuming the causes were external, you can requeue them using `rqx requeue <queue_name> --all`. Refer to `pychunkedgraph/ingest/cli.py` and `pychunkedgraph/ingest/rq_cli.py` for more commands.

> NOTE: make sure to flush redis (`ingest flush_redis`) after running ingest and before another `helm install`. Residuals from previous ingest runs can lead to inaccurate information.