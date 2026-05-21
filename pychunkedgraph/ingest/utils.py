# pylint: disable=invalid-name, missing-docstring

import functools
import math
import sys
from os import environ
from time import sleep
from typing import Dict, Generator, Tuple

import numpy as np
from kvdbclient import BigTableConfig, HBaseConfig
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rq import Queue, Retry
from rq.worker_registration import WORKERS_BY_QUEUE_KEY

from pychunkedgraph import get_logger

from . import IngestConfig
from .manager import IngestionManager
from ..graph import BackendClientInfo
from ..graph.meta import ChunkedGraphMeta, DataSource, GraphConfig
from ..graph.ocdbt import OcdbtConfig
from ..utils.general import chunked
from ..utils.redis import get_redis_connection
from ..utils.redis import keys as r_keys

logger = get_logger(__name__)

chunk_id_str = lambda layer, coords: f"{layer}_{'_'.join(map(str, coords))}"


def bootstrap(
    graph_id: str,
    config: dict,
    raw: bool = False,
    test_run: bool = False,
) -> Tuple[ChunkedGraphMeta, IngestConfig, BackendClientInfo, Dict]:
    """Parse config loaded from a yaml file.

    Returns ``(meta, ingest_config, client_info, ocdbt_config_dict)`` where the
    ocdbt config dict is sanitized through ``OcdbtConfig.from_dict(...).to_dict()``
    so unknown yaml keys are dropped and missing fields take dataclass defaults.
    """
    ingest_config = IngestConfig(
        **config.get("ingest_config", {}),
        USE_RAW_EDGES=raw,
        USE_RAW_COMPONENTS=raw,
        TEST_RUN=test_run,
    )
    backend_type = config["backend_client"].get("TYPE", "bigtable")
    if backend_type == "hbase":
        client_config = HBaseConfig(**config["backend_client"]["CONFIG"])
    else:
        client_config = BigTableConfig(**config["backend_client"]["CONFIG"])
    client_info = BackendClientInfo(backend_type, client_config)

    graph_config = GraphConfig(
        ID=f"{graph_id}",
        OVERWRITE=False,
        **config["graph_config"],
    )
    data_source = DataSource(**config["data_source"])

    meta = ChunkedGraphMeta(graph_config, data_source)
    ocdbt_config_dict = OcdbtConfig.from_dict(config.get("ocdbt_config")).to_dict()
    return (meta, ingest_config, client_info, ocdbt_config_dict)


def move_up(lines: int = 1):
    sys.stdout.write(f"\033[{lines}A")


def postprocess_edge_data(im, edge_dict):
    data_version = im.cg_meta.data_source.DATA_VERSION
    if data_version == 2:
        return edge_dict
    elif data_version in [3, 4]:
        new_edge_dict = {}
        for k in edge_dict:
            new_edge_dict[k] = {}
            if edge_dict[k] is None or len(edge_dict[k]) == 0:
                continue

            areas = (
                edge_dict[k]["area_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["area_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["area_z"] * im.cg_meta.resolution[2]
            )

            affs = (
                edge_dict[k]["aff_x"] * im.cg_meta.resolution[0]
                + edge_dict[k]["aff_y"] * im.cg_meta.resolution[1]
                + edge_dict[k]["aff_z"] * im.cg_meta.resolution[2]
            )

            new_edge_dict[k]["sv1"] = edge_dict[k]["sv1"]
            new_edge_dict[k]["sv2"] = edge_dict[k]["sv2"]
            new_edge_dict[k]["area"] = areas
            new_edge_dict[k]["aff"] = affs

        return new_edge_dict
    else:
        raise ValueError(f"Unknown data_version: {data_version}")


def randomize_grid_points(X: int, Y: int, Z: int) -> Generator[int, int, int]:
    indices = np.arange(X * Y * Z)
    np.random.shuffle(indices)
    for index in indices:
        yield np.unravel_index(index, (X, Y, Z))


def get_chunks_not_done(
    imanager: IngestionManager, layer: int, coords: list, splits: int = 0
) -> list:
    """check for set membership in redis in batches"""
    coords_strs = []
    if splits > 0:
        split_coords = []
        for coord in coords:
            for split in range(splits):
                jid = "_".join(map(str, coord)) + f"_{split}"
                coords_strs.append(jid)
                split_coords.append((coord, split))
    else:
        coords_strs = ["_".join(map(str, coord)) for coord in coords]
    try:
        completed = imanager.redis.smismember(f"{layer}c", coords_strs)
    except Exception:
        return split_coords if splits > 0 else coords

    if splits > 0:
        return [coord for coord, c in zip(split_coords, completed) if not c]
    return [coord for coord, c in zip(coords, completed) if not c]


def print_completion_rate(imanager: IngestionManager, layer: int, span: int = 30):
    rate = 0.0
    while True:
        counts = []
        print(f"{rate} chunks per second.")
        for _ in range(span + 1):
            counts.append(imanager.redis.scard(f"{layer}c"))
            sleep(1)
        rate = np.diff(counts).sum() / span
        move_up()


def _workers_busy_per_queue(redis, worker_keys_per_layer):
    """For each layer's set of worker keys, return parallel (workers, busy)
    string lists — "-" / "-" when no workers are registered for that layer.

    Two-round-trip approach: caller already fetched the SMEMBERS sets; this
    function pipelines HGET state for every worker key and counts busy.
    """
    state_pipe = redis.pipeline()
    for keys in worker_keys_per_layer:
        for wk in keys:
            state_pipe.hget(wk, "state")
    states = state_pipe.execute() if any(worker_keys_per_layer) else []

    workers, busy = [], []
    idx = 0
    for keys in worker_keys_per_layer:
        total = len(keys)
        b = 0
        for _ in keys:
            if states[idx] == b"busy":
                b += 1
            idx += 1
        workers.append(f"{total}" if total else "-")
        busy.append(f"{b}" if total else "-")
    return workers, busy


def _layer_keys(layers) -> list:
    """Stable per-layer redis keys (completed-set, queue list, failed zset, workers set).

    Returned once before the refresh loop so each refresh skips Queue /
    FailedJobRegistry construction and the lazy rq.registry import.
    """
    return [
        (
            f"{layer}c",
            f"rq:queue:l{layer}",
            f"rq:failed:l{layer}",
            WORKERS_BY_QUEUE_KEY % f"l{layer}",
        )
        for layer in layers
    ]


def _layer_status(redis, layer_keys):
    """Pipelined fetch of job_type + per-layer counts + busy-worker ratios."""
    pipeline = redis.pipeline()
    pipeline.get(r_keys.JOB_TYPE)
    for completed_key, queue_key, failed_key, workers_key in layer_keys:
        pipeline.scard(completed_key)
        pipeline.llen(queue_key)
        pipeline.zcard(failed_key)
        pipeline.smembers(workers_key)
    results = pipeline.execute()

    job_type = results[0].decode() if results[0] else "not_available"
    completed, queued, failed, worker_keys_per_layer = [], [], [], []
    for i in range(1, len(results), 4):
        completed.append(results[i])
        queued.append(results[i + 1])
        failed.append(results[i + 2])
        worker_keys_per_layer.append(results[i + 3])

    workers, busy = _workers_busy_per_queue(redis, worker_keys_per_layer)
    return job_type, completed, queued, failed, workers, busy


def _sized_table(columns: list, rows: list, **table_kwargs) -> Table:
    """Build a Rich Table whose column widths are sized to the actual data.

    `columns` is a list of (name, justify) tuples.
    `rows` is a list of tuples of cell strings (one per column).
    Each column gets width = max(len(name), max(len(cell)) over rows) so Rich
    never wraps or crops because no column is implicitly squeezed.
    """
    table = Table(
        box=None,
        pad_edge=False,
        padding=(0, 2),
        show_header=True,
        header_style="bold",
        **table_kwargs,
    )
    for col_idx, (name, justify) in enumerate(columns):
        width = max(len(name), max((len(row[col_idx]) for row in rows), default=0))
        # Header wrapped in Text so any brackets in `name` render literally
        # rather than being parsed as Rich markup tags.
        table.add_column(
            Text(name, style="bold"), justify=justify, width=width, no_wrap=True
        )
    for row in rows:
        table.add_row(*row)
    return table


def _aligned_kv_table(pairs: list, widths: list) -> Table:
    """One-data-row mini-table with externally-provided per-column widths."""
    table = Table(
        box=None, pad_edge=False, padding=(0, 1), show_header=True, header_style="bold"
    )
    for (name, _), w in zip(pairs, widths):
        table.add_column(name, justify="left", width=w, no_wrap=True)
    table.add_row(*(v for _, v in pairs))
    return table


def _header_renderables(imanager: IngestionManager) -> list:
    """Graph and ocdbt rows as mini-tables sharing column widths so columns line up."""
    graph_pairs = [
        ("version", str(imanager.cg.version)),
        ("graph_id", imanager.cg.graph_id),
        ("chunk_size", str(imanager.cg.meta.graph_config.CHUNK_SIZE)),
    ]
    ocdbt_pairs = []
    if imanager.ocdbt_seg:
        ocdbt_pairs = [
            ("ocdbt", str(imanager.ocdbt_seg)),
            ("populate_base", str(imanager.ocdbt_populate_base)),
            ("populate_layer", str(imanager.ocdbt_populate_layer)),
        ]

    # Per-column width = max length seen in EITHER row's header or value at that index.
    n = max(len(graph_pairs), len(ocdbt_pairs))
    widths = []
    for i in range(n):
        sizes = []
        if i < len(graph_pairs):
            sizes.append(len(graph_pairs[i][0]))
            sizes.append(len(graph_pairs[i][1]))
        if i < len(ocdbt_pairs):
            sizes.append(len(ocdbt_pairs[i][0]))
            sizes.append(len(ocdbt_pairs[i][1]))
        widths.append(max(sizes))

    out = [_aligned_kv_table(graph_pairs, widths)]
    if ocdbt_pairs:
        out.append(Rule(style="dim"))
        out.append(_aligned_kv_table(ocdbt_pairs, widths))
    return out


def _status_table(
    layers, layer_counts, completed, queued, failed, workers, busy
) -> Table:
    """One row per layer with progress, queue, and worker stats."""
    columns = [
        ("layer", "center"),
        ("queued", "right"),
        ("completed", "right"),
        ("total", "right"),
        ("progress", "right"),
        ("failed", "right"),
        ("workers", "right"),
        ("busy", "right"),
    ]
    rows = []
    for layer, done, count, q, f, w, b in zip(
        layers, completed, layer_counts, queued, failed, workers, busy
    ):
        pct = math.floor((done / count) * 100) if count else 0
        rows.append(
            (
                str(layer),
                f"{q:,}",
                f"{done:,}",
                f"{count:,}",
                f"{pct}%",
                f"{f:,}",
                str(w),
                str(b),
            )
        )
    return _sized_table(columns, rows)


def _status_renderable(
    imanager,
    layers,
    layer_counts,
    job_type,
    completed,
    queued,
    failed,
    workers,
    busy,
):
    """Combine header rows + per-layer table inside one Panel; job_type goes in the title."""
    body = Group(
        *_header_renderables(imanager),
        Rule(style="dim"),
        _status_table(layers, layer_counts, completed, queued, failed, workers, busy),
    )
    return Panel(
        body,
        title=job_type,
        title_align="left",
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def print_status(
    imanager: IngestionManager,
    redis,
    upgrade: bool = False,
    refresh_seconds: int = 5,
):
    """
    Print status to console.
    If `upgrade=True`, status does not include the root layer,
    since there is no need to update cross edges for root ids.
    `refresh_seconds` is how often redis is re-polled between redraws.
    """
    layers = range(2, imanager.cg_meta.layer_count + 1)
    if upgrade:
        layers = range(2, imanager.cg_meta.layer_count)
    layer_counts = imanager.cg_meta.layer_chunk_counts
    layer_keys = _layer_keys(layers)

    def render():
        return _status_renderable(
            imanager, layers, layer_counts, *_layer_status(redis, layer_keys)
        )

    # Start Live with a placeholder so the panel paints instantly; the first
    # real fetch (which includes redis connection setup) replaces it.
    with Live(Text("loading…"), screen=False) as live:
        while True:
            live.update(render())
            sleep(refresh_seconds)


def queue_layer_helper(
    parent_layer: int, imanager: IngestionManager, fn, splits: int = 0
):
    if parent_layer == imanager.cg_meta.layer_count:
        chunk_coords = [(0, 0, 0)]
    else:
        bounds = imanager.cg_meta.layer_chunk_bounds[parent_layer]
        chunk_coords = randomize_grid_points(*bounds)

    q = imanager.get_task_queue(f"l{parent_layer}")
    batch_size = int(environ.get("JOB_BATCH_SIZE", 10000))
    timeout_scale = int(environ.get("TIMEOUT_SCALE_FACTOR", 1))
    batches = chunked(chunk_coords, batch_size)
    failure_ttl = int(environ.get("FAILURE_TTL", 300))
    retry = int(environ.get("RETRY_COUNT", 0))
    max_queue_size = int(environ.get("QUEUE_SIZE", 100000))
    for batch in batches:
        _coords = get_chunks_not_done(imanager, parent_layer, batch, splits=splits)
        # buffer for optimal use of redis memory
        while len(q) > max_queue_size:
            logger.note(
                f"Queue has {len(q)} items (limit {max_queue_size}), waiting..."
            )
            sleep(10)

        job_datas = []
        for chunk_coord in _coords:
            if splits > 0:
                coord, split = chunk_coord
                jid = chunk_id_str(parent_layer, coord) + f"_{split}"
                job_datas.append(
                    Queue.prepare_data(
                        fn,
                        args=(parent_layer, coord, split, splits),
                        result_ttl=0,
                        job_id=jid,
                        timeout=f"{timeout_scale * int(parent_layer * parent_layer)}m",
                        retry=Retry(retry) if retry > 1 else None,
                        description="",
                        failure_ttl=failure_ttl,
                    )
                )
            else:
                job_datas.append(
                    Queue.prepare_data(
                        fn,
                        args=(parent_layer, chunk_coord),
                        result_ttl=0,
                        job_id=chunk_id_str(parent_layer, chunk_coord),
                        timeout=f"{timeout_scale * int(parent_layer * parent_layer)}m",
                        retry=Retry(retry) if retry > 1 else None,
                        description="",
                        failure_ttl=failure_ttl,
                    )
                )
        q.enqueue_many(job_datas)
        logger.note(f"Queued {len(job_datas)} chunks.")


def requeue_chunk(queue_name: str, chunk_info, atomic_fn, parent_fn):
    """Body of the ``chunk`` CLI command (shared by ingest and upgrade).

    Loads the manager from Redis, dispatches ``atomic_fn`` for L2 or
    ``parent_fn`` for L3+, and enqueues a single task with the standard
    job_id / timeout convention.
    """
    redis = get_redis_connection()
    imanager = IngestionManager.from_pickle(redis.get(r_keys.INGESTION_MANAGER))
    layer, coords = chunk_info[0], chunk_info[1:]
    if layer == 2:
        fn, args = atomic_fn, (coords,)
    else:
        fn, args = parent_fn, (layer, coords)
    queue = imanager.get_task_queue(queue_name)
    queue.enqueue(
        fn,
        job_id=chunk_id_str(layer, coords),
        job_timeout=f"{int(layer * layer)}m",
        result_ttl=0,
        args=args,
    )


def job_type_guard(job_type: str):
    def decorator_job_type_guard(func):
        @functools.wraps(func)
        def wrapper_job_type_guard(*args, **kwargs):
            redis = get_redis_connection()
            current_type = redis.get(r_keys.JOB_TYPE)
            if current_type is not None:
                current_type = current_type.decode()
                msg = (
                    f"Currently running `{current_type}`. You're attempting to run `{job_type}`."
                    f"\nRun `[flask] {current_type} flush_redis` to clear the current job and restart."
                )
                if current_type != job_type:
                    print(f"\n*WARNING*\n{msg}")
                    exit(1)
            return func(*args, **kwargs)

        return wrapper_job_type_guard

    return decorator_job_type_guard
