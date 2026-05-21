# tensorstore OCDBT reference

Every entry below was verified by probing tensorstore directly (intentional-bad-value + spec round-trip) against the binary in this workspace's venv. Re-verify if the tensorstore version changes.

## OCDBT kvstore spec — top-level fields

Sibling of `driver: "ocdbt"`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `base` | kvstore spec or URL | — | underlying kvstore (gcs/file/s3/…) |
| `manifest` | kvstore spec or URL | (under `base`) | the manifest *can* live in a separate kvstore from data |
| `config` | object | `{}` | see Config sub-fields below |
| `assume_config` | bool | `false` | skip reading the existing config from the manifest (use with care) |
| `coordinator` | ocdbt_coordinator resource | named ref `"ocdbt_coordinator"` | enables distributed mode when set |
| `cache_pool` | cache_pool resource | named ref `"cache_pool"` | |
| `data_copy_concurrency` | data_copy_concurrency resource | named ref | |
| `target_data_file_size` | uint64 | driver default | when a single commit's d/ writes exceed this, the writer rolls a new d/ file |
| `experimental_read_coalescing_threshold_bytes` | uint64 | — | |
| `experimental_read_coalescing_merged_bytes` | uint64 | — | |
| `experimental_read_coalescing_interval` | uint64 | — | |
| `btree_node_data_prefix` | string | `"d/"` | path prefix for btree-node files |
| `value_data_prefix` | string | `"d/"` | path prefix for value files |
| `version_tree_node_data_prefix` | string | `"d/"` | path prefix for version-tree files |
| `path` | string | `""` | sub-prefix in the kvstore |

**Not fields**: `data_file_prefixes`, `version_spec`, `recheck_cached*`, `transaction`, `btree_writer_concurrency`, `manifest_kind` (lives under `config`).

## OCDBT `config` sub-fields

| Field | Type | tensorstore default | Notes |
|---|---|---|---|
| `compression` | object | `{}` (none) | `{"id": "zstd", "level": N}` — zstd level 1–22 |
| `max_inline_value_bytes` | uint64 | `100` | values ≤ this size live inline in the btree leaf bytes; larger values get written to a d/ file and the mutation carries only an `IndirectDataReference`. In distributed mode this **directly bounds cooperator-forwarded RPC size**: inline values are carried inside the `WriteRequest.mutations` field, so a leaf's batch blows past the 4 MiB gRPC max-receive whenever multiple inline values pile up on one node. Source: `distributed/btree_writer.cc` `StagePending`. Setting low (≤ a few KB) pushes chunk values out-of-line → small mutations → small RPCs. |
| `max_decoded_node_bytes` | uint64 | `8388608` (8 MiB) | btree node split threshold. Larger nodes → shallower tree → fewer per-commit node touches. Setting this *smaller* than the default INCREASES per-commit forwarded bytes — empirically went from ~8 MiB to ~23 MiB RPCs when set to 1 MiB. |
| `version_tree_arity_log2` | int | — | controls version tree branching; rarely tuned |
| `manifest_kind` | enum | `"single"` | `"single"` or `"numbered"` (manifest history retained — needed for time-travel reads) |
| `uuid` | string | (auto) | 32-hex per-base UUID assigned at create time |

**Not fields**: `data_file_prefixes`, `data_file_prefix`, `btree_node_arity_log2`, `version_tree_node_arity`.

## `ocdbt_coordinator` context resource

| Field | Type | Default | Notes |
|---|---|---|---|
| `address` | string | — | `"host:port"` of the DistributedCoordinatorServer |
| `lease_duration` | duration string (`"1s"`, `"500ms"`, etc.) | — | how long a lease holder owns a btree node |
| `security` | object | `{method: "insecure"}` | requires `method` key. This build has **no** security methods registered (build flag) — all calls cleartext. |

## `DistributedCoordinatorServer({...})`

| Field | Type | Default | Notes |
|---|---|---|---|
| `bind_addresses` | list[string] | one ephemeral port | gRPC server bind address(es). `.port` after construction gives the ephemeral port. |
| `security` | object | insecure | same shape as the resource's security |

**There is NO Python knob for the gRPC server's max-receive message size.** The 4 MiB default is set inside tensorstore's gRPC server builder. Confirmed by strings on the binary: no `TENSORSTORE_*` env var, no spec/resource field, no Context resource that maps to `grpc.max_receive_message_length`.

## Distributed vs non-distributed write paths

The OCDBT driver picks one of two compiled implementations at open time:

- **non-distributed** (`btree_writer.cc`): coordinator absent. Each commit writes the manifest itself. Concurrent writers race the manifest CAS; losers retry; their pre-commit d/ writes become orphans.
- **distributed** (`distributed/btree_writer.cc`, `cooperator_*.cc`): coordinator present. One lease holder per btree node serializes commits. Other cooperators **forward their mutations over gRPC** to the lease holder.

### Constraints unique to distributed mode

1. **`ts.Transaction(atomic=True)` is incompatible.** "Cannot read/write … as single atomic transaction" — verified on (info + chunk) and on (cross-key). A plain `ts.Transaction()` still batches all writes into one OCDBT commit; only the *atomicity* across keys is lost.
2. **Cooperator-forwarded RPC ≤ ~4 MiB.** Carries (btree node delta) + (value bytes for keys committed into that node).
3. **Disjoint user-key writes still trigger forwarding.** Leases are per btree node, not per user-key range. Two workers writing distinct keys into the same node → one forwards to the other.

## Cooperator batching

`cooperator_submit_mutation_batch.cc` `SendToPeer` is the gRPC sender. The `WriteRequest` proto has `repeated bytes mutations` — each entry is one encoded `BtreeNodeWriteMutation` destined for the same leaf. The encoded mutation embeds the value_reference inline if it's an `absl::Cord`, or carries just an `IndirectDataReference` (small struct) otherwise. So **what's actually on the wire per RPC = (small request header) + Σ encoded mutations**, and each encoded mutation's size is dominated by its value bytes IF the value is inline.

Threshold for inline-vs-ref is `max_inline_value_bytes` (see config table). That's the real lever for RPC size.

What changes RPC size (verified by production dumps):
- `max_inline_value_bytes=1 MiB`, default node bytes → RPCs 5–8 MiB (inline chunks pile up in the batch)
- `max_inline_value_bytes=1 MiB` + `max_decoded_node_bytes=1 MiB` → RPCs up to 23 MiB (smaller nodes ≠ smaller RPCs)
- `max_inline_value_bytes=1 MiB` + dst `chunk_size` halved → RPCs grew to 12 MiB (more mutations per node → bigger batches)
- `max_inline_value_bytes=4 KiB` (chunks go out-of-line) → mutations carry only refs; RPC = small header + N×(key + ref + generation) → fits 4 MiB regardless of value sizes (this is the path our code takes)

## Defaults visible from spec round-trip

```json
{
  "assume_config": false,
  "btree_node_data_prefix": "d/",
  "config": {},
  "coordinator": "ocdbt_coordinator",
  "cache_pool": "cache_pool",
  "data_copy_concurrency": "data_copy_concurrency",
  "value_data_prefix": "d/",
  "version_tree_node_data_prefix": "d/"
}
```

## Env vars

- `OCDBT_COORDINATOR_HOST`, `OCDBT_COORDINATOR_PORT`: **NO EFFECT**. Not referenced anywhere in the binary. Address must go in spec's `coordinator.address`.
- `TENSORSTORE_VERBOSE_LOGGING`: comma-separated tag list to stderr. Tags include `ocdbt`, `coordinator`.

Other `TENSORSTORE_*` vars exist (CA paths, S3/GCS concurrency, etc.) — grep the binary.

## On-disk layout

- `manifest.ocdbt` at the base — root btree node + current data file refs.
- `d/` — directory of "data files" each holding concatenated values + (optionally) btree node bytes + version-tree node bytes.
- Each commit creates **at least one** d/ file holding all values + nodes for that commit, then a CAS-update of `manifest.ocdbt`.
- `target_data_file_size` controls when a single commit splits its d/ writes across files.

## How this maps onto pychunkedgraph

- `OcdbtConfig` (`pychunkedgraph/graph/ocdbt/meta.py`) → `compression: zstd 12`, `max_inline_value_bytes = 4 KiB`. The 4 KiB threshold keeps small metadata (info JSON, populate markers) inline while forcing every chunk value out-of-line into d/ files — this is what keeps cooperator RPCs under the 4 MiB gRPC ceiling.
- `create_base_ocdbt` / `open_base_ocdbt` pass `config.ts_config()` so the same OCDBT config persists across opens.
- `populate_chunk` (`pychunkedgraph/ingest/ocdbt.py`) opens the base with `coordinator_address` (distributed mode).
- `copy_ws_bbox_multiscale` uses **non-atomic** `ts.Transaction()` because of the distributed-mode constraint above.
- `_dump_failure_to_gcs` writes JSON failure forensics when `ERROR_DUMP` env is set.

## Empirically tried and ruled out

- `OCDBT_COORDINATOR_HOST/PORT` env vars — no effect.
- Bumping gRPC max-receive via env / channel arg / spec field — no such knob.
- Smaller `dst chunk_size` alone — RPC size grew (more mutations per node).
- Smaller `max_decoded_node_bytes` alone — RPC size grew (more per-commit node touches).
- `--ocdbt-edges` legacy path — decommissioned, removed.
- `ts.Transaction(atomic=True)` with distributed coordinator — incompatible.

## Open observations (not verified at production scale)

- `lease_duration` may reduce cross-cooperator forwarding if held long enough that a worker's whole task lands on its own nodes.
- `target_data_file_size` may affect manifest growth but not RPC size.
- Switching dst encoding from `compressed_segmentation` to `raw` would make per-value size predictable (`chunk_volume × bytes_per_voxel`), bypassing the dense-region pathological CS encoding (one observed key encoded to 23 MiB at 256×256×64).
