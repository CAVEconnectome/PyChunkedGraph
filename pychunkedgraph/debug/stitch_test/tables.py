import os, time

import cloudfiles.secrets
from google.cloud.bigtable import Client
from google.cloud.bigtable.backup import Backup

BACKUP_ID = "hsmith-mec-100gvx-exp16-0.26-backup"
CLUSTER_ID = "pychunkedgraph-c1"
PROJECT = "zetta-proofreading"
INSTANCE = "pychunkedgraph"
PREFIX = "stitch_redesign_test_"
EDGES_SRC = "gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges"

# suppress cloudfiles "Using default Google credentials" warning
cloudfiles.secrets.GOOGLE_CREDENTIALS_CACHE[""] = (PROJECT, None)


def setup_env():
    """Set env vars for BigTable access. Idempotent."""
    os.environ.setdefault("BIGTABLE_PROJECT", PROJECT)
    os.environ.setdefault("BIGTABLE_INSTANCE", INSTANCE)


def _get_instance():
    setup_env()
    client = Client(project=PROJECT, admin=True)
    return client.instance(INSTANCE)


def restore_test_table(table_name: str) -> str:
    """
    Restore a fresh copy from the hsmith_mec backup.
    Deletes existing table with the same name if present.
    Returns the table name once restore is complete.
    """
    if not table_name.startswith(PREFIX):
        raise ValueError(f"table name must start with '{PREFIX}', got '{table_name}'")

    instance = _get_instance()
    table = instance.table(table_name)
    if table.exists():
        print(f"deleting existing table {table_name}")
        table.delete()

    backup = Backup(BACKUP_ID, instance, cluster_id=CLUSTER_ID)
    print(f"restoring {table_name} from backup {BACKUP_ID}...")
    op = backup.restore(table_name)
    op.result()
    print(f"restored {table_name}")
    time.sleep(10)
    return table_name


def set_autoscaling(target_cpu=None, min_nodes=None):
    """Set autoscaling CPU target and/or min nodes. Returns previous values."""
    instance = _get_instance()
    clusters, _ = instance.list_clusters()
    cluster = clusters[0]
    prev_cpu = cluster.cpu_utilization_percent
    prev_min = cluster.min_serve_nodes
    changed = False
    if target_cpu is not None and prev_cpu != target_cpu:
        cluster.cpu_utilization_percent = target_cpu
        changed = True
    if min_nodes is not None and prev_min != min_nodes:
        cluster.min_serve_nodes = min_nodes
        changed = True
    if changed:
        cluster.update()
        parts = []
        if target_cpu is not None and prev_cpu != target_cpu:
            parts.append(f"cpu: {prev_cpu}% → {target_cpu}%")
        if min_nodes is not None and prev_min != min_nodes:
            parts.append(f"min_nodes: {prev_min} → {min_nodes}")
        print(f"autoscaling: {', '.join(parts)}")
    return prev_cpu, prev_min


def set_autoscaling_cpu(target_pct):
    """Convenience wrapper for backwards compatibility."""
    prev_cpu, _ = set_autoscaling(target_cpu=target_pct)
    return prev_cpu
