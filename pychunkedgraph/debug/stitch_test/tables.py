import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cloudfiles.secrets
from google.cloud.bigtable import Client
from google.cloud.bigtable.backup import Backup
from google.cloud.bigtable.data import BigtableDataClient, ReadRowsQuery, RowRange
from google.cloud.bigtable.data.row_filters import CellsRowLimitFilter, RowFilterChain, RowSampleFilter, StripValueTransformerFilter

BACKUP_ID = "hsmith-mec-100gvx-exp16-0.26-backup"
CLUSTER_ID = "pychunkedgraph-c1"
PROJECT = "zetta-proofreading"
INSTANCE = "pychunkedgraph"
PREFIX = "stitch_redesign_test_"
EDGES_SRC = "gs://dodam_exp/hammerschmith_mec/100GVx_cutout/proofreadable_exp16_0.26/agg_chunk_ext_edges"

_STRIP_FILTER = RowFilterChain(filters=[
    CellsRowLimitFilter(1),
    StripValueTransformerFilter(True),
])

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

    warm_cache(table_name)
    return table_name


WARM_RANDOM = True
WARM_SAMPLE_RATE = 0.1
WARM_ROWS_PER_TABLET = 2000


def warm_cache(table_name: str) -> None:
    """Scatter-read across tablets to prime BigTable block cache after restore.
    Two strategies via WARM_RANDOM:
      True  — RowSampleFilter scan (random block distribution)
      False — first N rows per tablet (sequential)
    All use CellsRowLimitFilter(1) + StripValueTransformerFilter to minimize transfer.
    One thread per tablet range for max parallelism.
    """
    instance = _get_instance()
    table = instance.table(table_name)

    samples = list(table.sample_row_keys())
    split_keys = [s.row_key for s in samples if s.row_key]
    total_bytes = samples[-1].offset_bytes if samples else 0
    print(f"  warm_cache: {len(split_keys)} tablet splits, ~{total_bytes / 1e9:.1f} GB, random={WARM_RANDOM}")

    data_client = BigtableDataClient(project=PROJECT)
    data_table = data_client.get_table(INSTANCE, table_name)

    ranges = []
    prev_key = b""
    for key in split_keys:
        ranges.append((prev_key, key))
        prev_key = key
    ranges.append((prev_key, b""))

    if WARM_RANDOM:
        row_filter = RowFilterChain(filters=[
            RowSampleFilter(WARM_SAMPLE_RATE),
            CellsRowLimitFilter(1),
            StripValueTransformerFilter(True),
        ])
        limit = None
    else:
        row_filter = _STRIP_FILTER
        limit = WARM_ROWS_PER_TABLET

    t0 = time.time()

    def _read_range(start_key: bytes, end_key: bytes) -> int:
        row_range = RowRange(
            start_key=start_key if start_key else None,
            end_key=end_key if end_key else None,
        )
        query = ReadRowsQuery(row_ranges=[row_range], row_filter=row_filter, limit=limit)
        return sum(1 for _ in data_table.read_rows(query))

    total_rows = 0
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {executor.submit(_read_range, s, e): (s, e) for s, e in ranges}
        for fut in as_completed(futures):
            total_rows += fut.result()

    data_client.close()
    print(f"  warm_cache: read {total_rows} rows across {len(ranges)} tablets in {time.time() - t0:.1f}s")


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
