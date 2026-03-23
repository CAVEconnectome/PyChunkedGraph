import os

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
    return table_name


def set_autoscaling_cpu(target_pct):
    """Set the autoscaling CPU utilization target. Returns the previous value."""
    instance = _get_instance()
    clusters, _ = instance.list_clusters()
    cluster = clusters[0]
    prev = cluster.cpu_utilization_percent
    if prev != target_pct:
        cluster.cpu_utilization_percent = target_pct
        cluster.update()
        print(f"autoscaling cpu: {prev}% → {target_pct}%")
    return prev
