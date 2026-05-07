# Hybrid base: precomputed + OCDBT fork (proposal)

Status: proposal, not implemented. Open question is whether storage and ingest-compute savings justify the read-path complexity.

## Problem

PCG ingest copies the entire watershed segmentation into `<ws>/ocdbt/base/` in OCDBT format before any CG edit can happen. Per-CG forks at `<ws>/ocdbt/<gid>/` store only the deltas from SV splits. Two costs follow:

- **Storage**: roughly 2× the segmentation footprint per dataset — original precomputed plus full OCDBT copy.
- **Ingest compute**: a per-chunk pass that reads the precomputed and writes it through the OCDBT driver. Hours of cluster time on TB-scale datasets.

Both costs are paid up-front, before any user has done a single edit. The proposal here: skip the base copy and serve unedited chunks directly from the raw precomputed directory. Per-CG OCDBT forks remain as the delta store.

## Why the current architecture has the base copy

Today's per-CG read spec is:

```
neuroglancer_precomputed
  └─ kvstore: ocdbt
       ├─ base: kvstack [base_layer, fork_manifest, fork_data]
       └─ config: { compression, max_inline_value_bytes, ... }
```

When a reader asks for chunk key `8_8_40/1024-..._0-128`:

1. The `neuroglancer_precomputed` driver passes the chunk key to its kvstore (the OCDBT driver).
2. OCDBT looks up the key **in its B+tree**. The B+tree's leaves map chunk keys to values.
3. If the key isn't in the B+tree, OCDBT returns not-found. It does not consult the kvstack any further.

The three kvstack layers serve OCDBT's *internal* storage (B+tree manifest + node blobs + leaf blobs) — they have no visibility into chunk-key lookups. So the OCDBT B+tree must contain every chunk key the reader will ever ask for, and that's why ingest copies the whole watershed: to populate the B+tree.

## What tensorstore primitives provide

Confirmed against tensorstore docs:

- **`kvstack` routes by exact / prefix match, with no fallthrough on miss.** A layer that claims a key range absorbs misses — they return `state='missing'` and do not cascade to the next layer. So we can't put raw precomputed below an OCDBT layer in a kvstack and expect kvstack to fall through when OCDBT doesn't have a key.
- **No native overlay/fallback kvstore driver.** `kvstack` is the only composition primitive at the kvstore level; it's precedence-based, not fallthrough.
- **OCDBT has no external-blob references.** B+tree leaves either inline the value or point to a data file under the OCDBT directory. There's no way to make a leaf reference a raw GCS precomputed file.
- **Array-level `stack` / `ts.overlay`** layers arrays by spatial domain. In overlapping regions, the later layer takes absolute precedence — missing-in-later does not fall back to earlier.

No single tensorstore primitive provides "try OCDBT delta first, fall through to raw precomputed on miss."

## Architectural options

### A — Two-stage read at the pcg layer

PCG reads open two handles: the OCDBT fork for the delta, and a raw `neuroglancer_precomputed` reader for the watershed base. For any voxel region, issue both reads and merge with "delta wins where present, base fills the rest."

- **Pros**: works inside pcg (`lookup_svs_from_seg`, sanity checks, debug tools) without any tensorstore changes.
- **Cons**: every pcg caller that uses `meta.ws_ocdbt` needs to route through a new merging reader. Neuroglancer doesn't benefit — it still gets a single kvstore spec from `dataset_info`. Either NG runs two layers itself (Option B) or we stand up a server-side proxy that does the merge before serving.

### B — NG-side layer stack

`dataset_info` publishes two precomputed layers: the raw watershed (read-only base) and the per-CG OCDBT fork (delta). NG composites them — visible segmentation is whichever has data at a given chunk.

- **Pros**: no change to pcg's read path. Pushes the architecture complexity into the viewer.
- **Cons**: requires NG to treat "missing chunk in delta" as "fall through to base," not "render as background." Default NG behavior is the latter, so a viewer-side or proxy-side shim is likely needed.

### C — Custom tensorstore kvstore driver

A new "fallthrough" kvstore driver: read tries layer N, falls through on miss to layer N−1. Implement upstream in tensorstore or fork-and-maintain.

- **Pros**: cleanest consumer-facing story — pcg and NG both keep using a single kvstore spec.
- **Cons**: tensorstore kvstore drivers are C++. Non-trivial maintenance surface; review/merge timeline if upstreaming.

### D — Lazy base population (not a win on its own)

Skip the ingest copy; copy a chunk from precomputed to OCDBT on first edit. Saves ingest compute. Does **not** save storage for reads — unedited chunks still 404 in OCDBT for a reader that doesn't have a fallback. Only useful in combination with A/B/C.

## Recommendation

Measure first. Confirm the actual storage and ingest-compute savings on a real dataset and weigh against the engineering cost of A/B/C.

If the savings justify the work, **A + B together** is the most pragmatic path:
- A gives pcg a single merged-read API. Edits, sanity checks, debug tooling keep working.
- B avoids standing up a proxy service for the viewer by letting NG handle the overlay.

Both require upstream verification:
- **For A**: confirm that `(x0:x1, y0:y1, z0:z1)` reads on an OCDBT with sparse keys surface missing-ness *per chunk* at the `neuroglancer_precomputed` array layer (not per-region, not silently fill-valued).
- **For B**: confirm NG's segmentation loader can be configured to fall through gaps in one layer to another. If it can't, build a small server-side merging shim — at which point Option A's reader becomes that shim and B reduces to "publish two specs."

C is the cleanest design but carries the highest cost. Pursue only if A/B turn out to have unworkable semantics.

## Open questions before any implementation

1. Does OCDBT's `read_result.state == 'missing'` surface per-chunk at the `neuroglancer_precomputed` array layer, or does the array silently fill missing chunks with fill-value? Verifiable by opening an OCDBT with sparse keys and reading a region that spans present + missing chunks.
2. Does NG distinguish "chunk returned as missing" from "chunk is all fill-value"? If not, a viewer-side overlay needs a shim regardless.
3. What's the actual delta volume per CG over its lifetime? If SV splits eventually touch a significant fraction of chunks, the storage win shrinks toward zero — at which point the simpler architecture (today's full base copy) wins on engineering cost.

## Files to start from when implementing

- `pychunkedgraph/graph/ocdbt.py` — spec construction (`build_cg_ocdbt_spec`), base population (`create_base_ocdbt`), fork setup (`fork_base_manifest`).
- `pychunkedgraph/ingest/cli.py`, `pychunkedgraph/ingest/cluster.py` — current base-copy flow.
- `pychunkedgraph/graph/utils/generic.py::get_local_segmentation` — single pcg read entry point that would need the two-stage merge in Option A.

## Verification (per chosen option)

- **A**: unit test that simulates a partial-delta OCDBT + raw precomputed and confirms the pcg reader returns the correct labels for spans crossing both.
- **B**: configure an NG link with both layers against a test dataset; compare the rendered segmentation to a known-good reference at edited and unedited regions.
- **C**: a tensorstore build with the new driver passes a fallthrough test (missing key in upper layer resolves from lower layer).
