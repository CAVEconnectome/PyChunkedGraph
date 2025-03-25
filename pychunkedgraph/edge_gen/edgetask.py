import json
import os
import re
import sys
from copy import deepcopy
from functools import lru_cache
from itertools import chain
from operator import itemgetter
from typing import Iterable, Mapping, Tuple, Union

from cloudvolume import CloudVolume, Storage

import MySQLdb
import numpy as np
import zstandard as zstd

sys.path.insert(0, os.path.join(sys.path[0], '..'))
from backend import chunkedgraph  # noqa

UINT64_ZERO = np.uint64(0)
UINT64_ONE = np.uint64(1)


class EdgeTask:
    def __init__(self,
                 cgraph: chunkedgraph.ChunkedGraph,
                 mysql_conn: any,
                 agglomeration_input: CloudVolume,
                 watershed_input: CloudVolume,
                 regiongraph_input: Storage,
                 regiongraph_output: Storage,
                 regiongraph_chunksize: Tuple[int, int, int],
                 roi: Tuple[slice, slice, slice]):
        self.__cgraph = cgraph
        self.__mysql_conn = mysql_conn
        self.__watershed = {
            "cv_input": watershed_input,
            "original": np.array([], dtype=np.uint64, ndmin=3),
            "relabeled": np.array([], dtype=np.uint64, ndmin=3),
            "rg2cg_complete": {},
            "rg2cg_boundary": {}
        }
        self.__agglomeration = {
            "cv": agglomeration_input,
            "original": np.array([], dtype=np.uint64, ndmin=3)
        }
        self.__regiongraph = {
            "storage_in": regiongraph_input,
            "storage_out": regiongraph_output,
            "edges": {},
            "chunksize": regiongraph_chunksize,
            "offset": self.__watershed["cv_input"].voxel_offset,
            "maxlevel": int(np.ceil(np.log2(np.max(np.floor_divide(
                self.__watershed["cv_input"].volume_size, regiongraph_chunksize)))))
        }
        self.__roi = roi
        self.__watershed["original"] = self.__watershed["cv_input"][self.__roi]
        self.__watershed["relabeled"] = np.empty_like(self.__watershed["original"])
        self.__agglomeration["original"] = \
            self.__agglomeration["cv"][self.__roi]

    def execute(self):
        self.__relabel_cutout()

        self.__compute_cutout_regiongraph()

        return

    def get_relabeled_watershed(self):
        return self.__watershed["relabeled"][0:-1, 0:-1, 0:-1, :]

    def __load_rg_chunkhierarchy_affinities(self):
        """
        Collect all weighted edges from the Region Graph chunk hierarchy
        within the ROI.
        """

        # Convert ROI (in voxel) to Region Graph chunk indices
        chunk_range = tuple(map(
            lambda x:
                np.floor_divide(
                    np.maximum(0, np.subtract(x, self.__regiongraph["offset"])),
                    self.__regiongraph["chunksize"]),
                ((self.__roi[0].start, self.__roi[1].start, self.__roi[2].start),
                 (self.__roi[0].stop, self.__roi[1].stop, self.__roi[2].stop))
        ))

        # TODO: Possible speedup by skipping high level chunks that don't
        #       intersect with ROI
        edges = []
        for l in range(self.__regiongraph["maxlevel"] + 1):
            for x in range(chunk_range[0][0], chunk_range[1][0] + 1):
                for y in range(chunk_range[0][1], chunk_range[1][1] + 1):
                    for z in range(chunk_range[0][2], chunk_range[1][2] + 1):
                        print("Loading layer %i: (%i,%i,%i)" % (l, x, y, z))
                        chunk_path = "edges_%i_%i_%i_%i.data.zst" % (l, x, y, z)
                        edges.append(load_rg_chunk_affinities(
                            self.__regiongraph["storage_in"], chunk_path)
                        )

            chunk_range = (chunk_range[0] // 2, chunk_range[1] // 2)

        print("Converting to Set")
        return {e.item()[0:2]: e for e in chain(*edges)}

    def __load_cutout_labels_from_db(self):
        chunks_to_fetch = []
        for x in range(self.__roi[0].start, self.__roi[0].stop, self.__cgraph.chunk_size[0]):
            for y in range(self.__roi[1].start, self.__roi[1].stop, self.__cgraph.chunk_size[1]):
                for z in range(self.__roi[2].start, self.__roi[2].stop, self.__cgraph.chunk_size[2]):
                    chunks_to_fetch.append(self.__cgraph.get_chunk_id_from_coord(1, x, y, z))

        self.__mysql_conn.query("SELECT id, edges FROM chunkedges WHERE id IN (%s);" % ",".join(str(x) for x in chunks_to_fetch))
        res = self.__mysql_conn.store_result()

        chunk_labels = {x: {} for x in chunks_to_fetch}
        for row in res.fetch_row(maxrows=0):
            edges_iter = iter(np.frombuffer(row[1], dtype=np.uint64))
            chunk_labels[row[0]] = dict(zip(edges_iter, edges_iter))

        self.__watershed["rg2cg_boundary"] = chunk_labels
        self.__watershed["rg2cg_complete"] = deepcopy(self.__watershed["rg2cg_boundary"])

    def __save_cutout_labels_to_db(self):
        self.__mysql_conn.query("START TRANSACTION;")

        chunk_labels = self.__watershed["rg2cg_boundary"]
        for chunk_id, mappings in chunk_labels.items():
            if len(mappings) > 0:
                flat_binary_mapping = np.fromiter((item for k in mappings for item in (k, mappings[k])), dtype=np.uint64).tobytes()
                flat_binary_mapping_escaped = self.__mysql_conn.escape_string(flat_binary_mapping)
                self.__mysql_conn.query(b"INSERT INTO chunkedges (id, edges) VALUES (%i, \"%s\") ON DUPLICATE KEY UPDATE edges = VALUES(edges);" % (chunk_id, flat_binary_mapping_escaped))

        self.__mysql_conn.query("COMMIT;")

    def __relabel_cutout(self):
        # Load existing labels of center + neighboring chunks
        self.__load_cutout_labels_from_db()
        assigned_node_ids = {node_id for chunk_edges in self.__watershed["rg2cg_complete"].values() for node_id in chunk_edges.values()}

        def relabel_chunk(chunk_id: np.uint64, view_range: Tuple[slice, slice, slice]):
            next_segment_id = UINT64_ONE

            original = np.nditer(
                self.__watershed["original"][view_range], flags=['multi_index'])
            relabeled = np.nditer(self.__watershed["relabeled"][view_range], flags=['multi_index'], op_flags=['writeonly'])

            print("Starting Loop for chunk %i" % chunk_id)
            while not original.finished:
                original_val = np.uint64(original[0])

                if original_val == UINT64_ZERO:
                    # Don't relabel cell boundary (ID 0)
                    relabeled[0] = UINT64_ZERO
                elif original_val in self.__watershed["rg2cg_complete"][chunk_id]:
                    # Already encountered this ID before.
                    relabeled[0] = relabeled_val = self.__watershed["rg2cg_complete"][chunk_id][original_val]
                    if original.multi_index[0] == 0 or \
                       original.multi_index[1] == 0 or \
                       original.multi_index[2] == 0:
                        self.__watershed["rg2cg_boundary"][chunk_id][original_val] = relabeled_val
                else:
                    # Find new, unused node ID for this chunk.
                    while self.__cgraph.get_node_id(
                            segment_id=next_segment_id,
                            chunk_id=chunk_id) in assigned_node_ids:
                        next_segment_id += UINT64_ONE

                    relabeled_val = self.__cgraph.get_node_id(
                            segment_id=next_segment_id,
                            chunk_id=chunk_id)

                    relabeled[0] = relabeled_val
                    next_segment_id += UINT64_ONE
                    assigned_node_ids.add(relabeled_val)

                    self.__watershed["rg2cg_complete"][chunk_id][original_val] = relabeled_val
                    if original.multi_index[0] == 0 or \
                       original.multi_index[1] == 0 or \
                       original.multi_index[2] == 0:
                        self.__watershed["rg2cg_boundary"][chunk_id][original_val] = relabeled_val

                original.iternext()
                relabeled.iternext()

        for x_start in (0, int(self.__cgraph.chunk_size[0])):
            for y_start in (0, int(self.__cgraph.chunk_size[1])):
                for z_start in (0, int(self.__cgraph.chunk_size[2])):
                    x_end = x_start + int(self.__cgraph.chunk_size[0])
                    y_end = y_start + int(self.__cgraph.chunk_size[1])
                    z_end = z_start + int(self.__cgraph.chunk_size[2])

                    chunk_id = self.__cgraph.get_chunk_id_from_coord(
                        layer=1,
                        x=self.__roi[0].start + x_start,
                        y=self.__roi[1].start + y_start,
                        z=self.__roi[2].start + z_start)

                    relabel_chunk(chunk_id, (slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)))

        self.__save_cutout_labels_to_db()

    def __compute_cutout_regiongraph(self):
        edges_center_connected = np.array([])
        edges_center_disconnected = np.array([])
        isolated_sv = np.array([])
        edges_xplus_connected = np.array([])
        edges_xplus_disconnected = np.array([])
        edges_xplus_unbreakable = np.array([])
        edges_yplus_connected = np.array([])
        edges_yplus_disconnected = np.array([])
        edges_yplus_unbreakable = np.array([])
        edges_zplus_connected = np.array([])
        edges_zplus_disconnected = np.array([])
        edges_zplus_unbreakable = np.array([])

        if np.any(self.__watershed["original"]):
            # Download all region graph edges covering this part of the dataset
            regiongraph_edges = self.__load_rg_chunkhierarchy_affinities()

            print("Calculating RegionGraph...")

            original = self.__watershed["original"]
            agglomeration = self.__agglomeration["original"]

            # Shortcut to Original -> Relabeled supervoxel lookup table for the
            # center chunk
            rg2cg_center = self.__watershed["rg2cg_complete"][
                self.__cgraph.get_chunk_id_from_coord(
                    layer=1,
                    x=self.__roi[0].start,
                    y=self.__roi[1].start,
                    z=self.__roi[2].start)]

            # Original -> Relabeled supervoxel lookup table for chunk in X+ dir
            rg2cg_xplus = self.__watershed["rg2cg_complete"][
                self.__cgraph.get_chunk_id_from_coord(
                    layer=1,
                    x=self.__roi[0].start + int(self.__cgraph.chunk_size[0]),
                    y=self.__roi[1].start,
                    z=self.__roi[2].start)]

            # Original -> Relabeled supervoxel lookup table for chunk in Y+ dir
            rg2cg_yplus = self.__watershed["rg2cg_complete"][
                self.__cgraph.get_chunk_id_from_coord(
                    layer=1,
                    x=self.__roi[0].start,
                    y=self.__roi[1].start + int(self.__cgraph.chunk_size[1]),
                    z=self.__roi[2].start)]

            # Original -> Relabeled supervoxel lookup table for chunk in Z+ dir
            rg2cg_zplus = self.__watershed["rg2cg_complete"][
                self.__cgraph.get_chunk_id_from_coord(
                    layer=1,
                    x=self.__roi[0].start,
                    y=self.__roi[1].start,
                    z=self.__roi[2].start + int(self.__cgraph.chunk_size[2]))]

            # Mask unsegmented voxel (ID=0) and voxel not at a supervoxel
            # boundary in X-direction
            sv_boundaries_x = \
                (original[:-1, :, :] != UINT64_ZERO) & (original[1:, :, :] != UINT64_ZERO) & \
                (original[:-1, :, :] != original[1:, :, :])

            # Mask voxel that are not at an agglomeration boundary in X-direction
            agg_boundaries_x = (agglomeration[:-1, :, :] == agglomeration[1:, :, :])

            # Mask unsegmented voxel (ID=0) and voxel not at a supervoxel
            # boundary in Y-direction
            sv_boundaries_y = \
                (original[:, :-1, :] != UINT64_ZERO) & (original[:, 1:, :] != UINT64_ZERO) & \
                (original[:, :-1, :] != original[:, 1:, :])

            # Mask voxel that are not at an agglomeration boundary in Y-direction
            agg_boundaries_y = (agglomeration[:, :-1, :] == agglomeration[:, 1:, :])

            # Mask unsegmented voxel (ID=0) and voxel not at a supervoxel
            # boundary in Z-direction
            sv_boundaries_z = \
                (original[:, :, :-1] != UINT64_ZERO) & (original[:, :, 1:] != UINT64_ZERO) & \
                (original[:, :, :-1] != original[:, :, 1:])

            # Mask voxel that are not at an agglomeration boundary in Z-direction
            agg_boundaries_z = (agglomeration[:, :, :-1] == agglomeration[:, :, 1:])

            # Center Chunk:
            # Collect all unique pairs of adjacent supervoxel IDs from the original
            # watershed labeling that are part of the same agglomeration.
            # Note that edges are sorted (lower supervoxel ID comes first).
            edges_center_connected = {x if x[0] < x[1] else (x[1], x[0]) for x in chain(
                zip(original[:-2, :-1, :-1][sv_boundaries_x[:-1, :-1, :-1] & agg_boundaries_x[:-1, :-1, :-1]],
                    original[1:-1, :-1, :-1][sv_boundaries_x[:-1, :-1, :-1] & agg_boundaries_x[:-1, :-1, :-1]]),
                zip(original[:-1, :-2, :-1][sv_boundaries_y[:-1, :-1, :-1] & agg_boundaries_y[:-1, :-1, :-1]],
                    original[:-1, 1:-1, :-1][sv_boundaries_y[:-1, :-1, :-1] & agg_boundaries_y[:-1, :-1, :-1]]),
                zip(original[:-1, :-1, :-2][sv_boundaries_z[:-1, :-1, :-1] & agg_boundaries_z[:-1, :-1, :-1]],
                    original[:-1, :-1, 1:-1][sv_boundaries_z[:-1, :-1, :-1] & agg_boundaries_z[:-1, :-1, :-1]]))}

            # Look up the affinity information for each edge and replace
            # original supervoxel IDs with relabeled IDs
            if edges_center_connected:
                edges_center_connected = np.array([
                    (*sorted(itemgetter(x[0], x[1])(rg2cg_center)), x[2], x[3])
                    for x in [regiongraph_edges[e] for e in edges_center_connected]
                ], dtype='uint64, uint64, float32, uint64')
            else:
                edges_center_connected = np.array([], dtype='uint64, uint64, float32, uint64')

            # Collect all unique pairs of adjacent supervoxel IDs from the original
            # watershed labeling that are NOT part of the same agglomeration.
            edges_center_disconnected = {x if x[0] < x[1] else (x[1], x[0]) for x in chain(
                zip(original[:-2, :-1, :-1][sv_boundaries_x[:-1, :-1, :-1] & ~agg_boundaries_x[:-1, :-1, :-1]],
                    original[1:-1, :-1, :-1][sv_boundaries_x[:-1, :-1, :-1] & ~agg_boundaries_x[:-1, :-1, :-1]]),
                zip(original[:-1, :-2, :-1][sv_boundaries_y[:-1, :-1, :-1] & ~agg_boundaries_y[:-1, :-1, :-1]],
                    original[:-1, 1:-1, :-1][sv_boundaries_y[:-1, :-1, :-1] & ~agg_boundaries_y[:-1, :-1, :-1]]),
                zip(original[:-1, :-1, :-2][sv_boundaries_z[:-1, :-1, :-1] & ~agg_boundaries_z[:-1, :-1, :-1]],
                    original[:-1, :-1, 1:-1][sv_boundaries_z[:-1, :-1, :-1] & ~agg_boundaries_z[:-1, :-1, :-1]]))}

            # Look up the affinity information for each edge and replace
            # original supervoxel IDs with relabeled IDs
            if edges_center_disconnected:
                edges_center_disconnected = np.array([
                    (*sorted(itemgetter(x[0], x[1])(rg2cg_center)), x[2], x[3])
                    for x in [regiongraph_edges[e] for e in edges_center_disconnected]
                ], dtype='uint64, uint64, float32, uint64')
            else:
                edges_center_disconnected = np.array([], dtype='uint64, uint64, float32, uint64')

            # Check if there are supervoxel that are not connected to any other
            # supervoxel - surrounded by ID 0
            isolated_sv = set(rg2cg_center.values())
            for e in chain(edges_center_connected, edges_center_disconnected):
                isolated_sv.discard(e[0])
                isolated_sv.discard(e[1])
            isolated_sv = np.array(list(isolated_sv), dtype=np.uint64)

            # XPlus Chunk:
            # Collect edges between center chunk and the chunk in X+ direction.
            # Slightly different approach because the relabeling lookup needs
            # to be done for two different dictionaries. Slower, but fast enough
            # due to far fewer edges near the boundary.
            # Node ID layout guarantees that center chunk IDs are always smaller
            # than IDs of positive neighboring chunks.
            edges_xplus_connected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_xplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[-2:-1, :-1, :-1][sv_boundaries_x[-1:, :-1, :-1] & agg_boundaries_x[-1:, :-1, :-1]],
                    original[-1:, :-1, :-1][sv_boundaries_x[-1:, :-1, :-1] & agg_boundaries_x[-1:, :-1, :-1]])
            }), dtype='uint64, uint64, float32, uint64')

            edges_xplus_disconnected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_xplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[-2:-1, :-1, :-1][sv_boundaries_x[-1:, :-1, :-1] & ~agg_boundaries_x[-1:, :-1, :-1]],
                    original[-1:, :-1, :-1][sv_boundaries_x[-1:, :-1, :-1] & ~agg_boundaries_x[-1:, :-1, :-1]])
            }), dtype='uint64, uint64, float32, uint64')

            # Unbreakable edges (caused by relabeling and chunking) don't have
            # sum of area or affinity values
            edges_xplus_unbreakable = np.array(list({
                (rg2cg_center[x], rg2cg_xplus[x]) for x in np.unique(
                    original[-2:-1, :-1, :-1][(original[-2:-1, :-1, :-1] != UINT64_ZERO) &
                                              (original[-2:-1, :-1, :-1] == original[-1:, :-1, :-1])])
            }), dtype='uint64, uint64')

            # YPlus Chunk:
            # Collect edges between center chunk and the chunk in Y+ direction.
            edges_yplus_connected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_yplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[:-1, -2:-1, :-1][sv_boundaries_y[:-1, -1:, :-1] & agg_boundaries_y[:-1, -1:, :-1]],
                    original[:-1, -1:, :-1][sv_boundaries_y[:-1, -1:, :-1] & agg_boundaries_y[:-1, -1:, :-1]])
            }), dtype='uint64, uint64, float32, uint64')

            edges_yplus_disconnected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_yplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[:-1, -2:-1, :-1][sv_boundaries_y[:-1, -1:, :-1] & ~agg_boundaries_y[:-1, -1:, :-1]],
                    original[:-1, -1:, :-1][sv_boundaries_y[:-1, -1:, :-1] & ~agg_boundaries_y[:-1, -1:, :-1]])
            }), dtype='uint64, uint64, float32, uint64')

            edges_yplus_unbreakable = np.array(list({
                (rg2cg_center[x], rg2cg_yplus[x]) for x in np.unique(
                    original[:-1, -2:-1, :-1][(original[:-1, -2:-1, :-1] != UINT64_ZERO) &
                                              (original[:-1, -2:-1, :-1] == original[:-1, -1:, :-1])])
            }), dtype='uint64, uint64')

            # ZPlus Chunk
            # Collect edges between center chunk and the chunk in Z+ direction.
            edges_zplus_connected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_zplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[:-1, :-1, -2:-1][sv_boundaries_z[:-1, :-1, -1:] & agg_boundaries_z[:-1, :-1, -1:]],
                    original[:-1, :-1, -1:][sv_boundaries_z[:-1, :-1, -1:] & agg_boundaries_z[:-1, :-1, -1:]])
            }), dtype='uint64, uint64, float32, uint64')

            edges_zplus_disconnected = np.array(list({
                (rg2cg_center[x[0]],
                 rg2cg_zplus[x[1]],
                 *regiongraph_edges[x if x[0] < x[1] else (x[1], x[0])].item()[2:]) for x in zip(
                    original[:-1, :-1, -2:-1][sv_boundaries_z[:-1, :-1, -1:] & ~agg_boundaries_z[:-1, :-1, -1:]],
                    original[:-1, :-1, -1:][sv_boundaries_z[:-1, :-1, -1:] & ~agg_boundaries_z[:-1, :-1, -1:]])
            }), dtype='uint64, uint64, float32, uint64')

            edges_zplus_unbreakable = np.array(list({
                (rg2cg_center[x], rg2cg_zplus[x]) for x in np.unique(
                    original[:-1, :-1, -2:-1][(original[:-1, :-1, -2:-1] != UINT64_ZERO) &
                                              (original[:-1, :-1, -2:-1] == original[:-1, :-1, -1:])])
            }), dtype='uint64, uint64')
        else:
            print("Fast skipping Regiongraph calculation - empty block")

        # Prepare upload
        rg2cg_center_str = slice_to_str(
            slice(self.__roi[x].start,
                  self.__roi[x].start + int(self.__cgraph.chunk_size[x])) for x in range(3))

        rg2cg_xplus_str = slice_to_str((
            slice(self.__roi[0].start + int(self.__cgraph.chunk_size[0]) - 1,
                  self.__roi[0].start + int(self.__cgraph.chunk_size[0]) + 1),
            slice(self.__roi[1].start, self.__roi[1].start + int(self.__cgraph.chunk_size[1])),
            slice(self.__roi[2].start, self.__roi[2].start + int(self.__cgraph.chunk_size[2]))))

        rg2cg_yplus_str = slice_to_str((
            slice(self.__roi[0].start, self.__roi[0].start + int(self.__cgraph.chunk_size[0])),
            slice(self.__roi[1].start + int(self.__cgraph.chunk_size[1]) - 1,
                  self.__roi[1].start + int(self.__cgraph.chunk_size[1]) + 1),
            slice(self.__roi[2].start, self.__roi[2].start + int(self.__cgraph.chunk_size[2]))))

        rg2cg_zplus_str = slice_to_str((
            slice(self.__roi[0].start, self.__roi[0].start + int(self.__cgraph.chunk_size[0])),
            slice(self.__roi[1].start, self.__roi[1].start + int(self.__cgraph.chunk_size[1])),
            slice(self.__roi[2].start + int(self.__cgraph.chunk_size[2]) - 1,
                  self.__roi[2].start + int(self.__cgraph.chunk_size[2]) + 1)))

        print("Uploading edges")
        self.__regiongraph["storage_out"].put_files(
            files=[(rg2cg_center_str + '_connected.bin', edges_center_connected.tobytes()),
                   (rg2cg_center_str + '_disconnected.bin', edges_center_disconnected.tobytes()),
                   (rg2cg_center_str + '_isolated.bin', isolated_sv.tobytes()),
                   (rg2cg_xplus_str + '_connected.bin', edges_xplus_connected.tobytes()),
                   (rg2cg_xplus_str + '_disconnected.bin', edges_xplus_disconnected.tobytes()),
                   (rg2cg_xplus_str + '_unbreakable.bin', edges_xplus_unbreakable.tobytes()),
                   (rg2cg_yplus_str + '_connected.bin', edges_yplus_connected.tobytes()),
                   (rg2cg_yplus_str + '_disconnected.bin', edges_yplus_disconnected.tobytes()),
                   (rg2cg_yplus_str + '_unbreakable.bin', edges_yplus_unbreakable.tobytes()),
                   (rg2cg_zplus_str + '_connected.bin', edges_zplus_connected.tobytes()),
                   (rg2cg_zplus_str + '_disconnected.bin', edges_zplus_disconnected.tobytes()),
                   (rg2cg_zplus_str + '_unbreakable.bin', edges_zplus_unbreakable.tobytes())],
            content_type='application/octet-stream')
        print("Done")


@lru_cache(maxsize=32)
def load_rg_chunk_affinities(regiongraph_storage: Storage, chunk_path: str) -> np.ndarray:
    """
    Extract weighted supervoxel edges from zstd compressed Region Graph
    file `chunk_path`.
    The unversioned, custom binary file format shall be called RanStruct,
    which, as of 2018-08-03, looks like this:

    struct RanStruct        # Little Endian, not aligned -> 56 Byte
      segA1::UInt64
      segB1::UInt64
      sum_aff1::Float32
      sum_area1::UInt64
      segA2::UInt64         # same as segA1
      segB2::UInt64         # same as segB1
      sum_aff2::Float32     # same as sum_aff1
      sum_area2::UInt64     # same as sum_area1
    end

    The big top level Region Graph chunks get requested almost every time,
    thus the memoization.
    """

    f = regiongraph_storage.get_file(chunk_path)
    if not f:
        Warning("%s doesn't exist" % chunk_path)
        return np.array([], dtype='uint64, uint64, float32, uint64')

    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(f)

    buf = np.frombuffer(decompressed, dtype='uint64, uint64, float32, uint64')
    return np.lib.stride_tricks.as_strided(
        buf,
        shape=tuple(x//2 for x in buf.shape),
        strides=tuple(x*2 for x in buf.strides),
        writeable=False
    )


def str_to_slice(slice_str: str) -> Tuple[slice, slice, slice]:
    match = re.match(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", slice_str)
    return (slice(int(match.group(1)), int(match.group(2))),
            slice(int(match.group(3)), int(match.group(4))),
            slice(int(match.group(5)), int(match.group(6))))


def slice_to_str(slices: Union[slice, Iterable[slice]]) -> str:
    if isinstance(slices, slice):
        return "%d-%d" % (slices.start, slices.stop)
    else:
        return '_'.join(map(slice_to_str, slices))


def run_task_bundle(settings: Mapping, roi: Tuple[slice, slice, slice]):
    # Remember: DB must be cleared before starting a whole new run
    with open("/secrets/mysql") as passwd:
        mysql_conn = MySQLdb.connect(
            host=settings["mysql"]["host"],
            user=settings["mysql"]["user"],
            db=settings["mysql"]["db"],
            passwd=passwd.read().strip()
        )

    cgraph = chunkedgraph.ChunkedGraph(
        table_id=settings["chunkedgraph"]["table_id"],
        instance_id=settings["chunkedgraph"]["instance_id"]
    )

    # Things to check:
    # - Agglomeration and Input Watershed have the same offset/size
    # - Taskbundle Offset and ROI is a multiple of cgraph.chunksize
    # - Output Watershed chunksize must be a multiple of cgraph.chunksize

    agglomeration_input = CloudVolume(
        settings["layers"]["agglomeration_path_input"], bounded=False)
    watershed_input = CloudVolume(
        settings["layers"]["watershed_path_input"], bounded=False)
    watershed_output = CloudVolume(
        settings["layers"]["watershed_path_output"], bounded=False, autocrop=True)
    regiongraph_input = Storage(
        settings["regiongraph"]["regiongraph_path_input"])
    regiongraph_output = Storage(
        settings["regiongraph"]["regiongraph_path_output"])
    regiongraph_chunksize = tuple(settings["regiongraph"]["chunksize"])

    chunkgraph_chunksize = np.array(cgraph.chunk_size, dtype=int)
    output_watershed_chunksize = np.array(watershed_output.underlying, dtype=int)
    outer_chunksize = np.maximum(chunkgraph_chunksize, output_watershed_chunksize, dtype=int)

    # Iterate through TaskBundle using a minimal chunk size that is a multiple
    # of the output watershed chunk size and the Chunked Graph chunk size.
    for ox in range(roi[0].start, roi[0].stop, outer_chunksize[0]):
        for oy in range(roi[1].start, roi[1].stop, outer_chunksize[1]):
            for oz in range(roi[2].start, roi[2].stop, outer_chunksize[2]):

                watershed_output_buffer = np.zeros((*outer_chunksize, 1), dtype=np.uint64)

                # Iterate through ChunkGraph chunk-sized tasks:
                for ix_start in range(0, outer_chunksize[0], chunkgraph_chunksize[0]):
                    for iy_start in range(0, outer_chunksize[1], chunkgraph_chunksize[1]):
                        for iz_start in range(0, outer_chunksize[2], chunkgraph_chunksize[2]):
                            ix_end = ix_start + chunkgraph_chunksize[0]
                            iy_end = iy_start + chunkgraph_chunksize[1]
                            iz_end = iz_start + chunkgraph_chunksize[2]

                            # One voxel overlap in each dimension to get
                            # consistent labeling across chunks
                            edgetask_roi = (slice(ox + ix_start, ox + ix_end + 1),
                                            slice(oy + iy_start, oy + iy_end + 1),
                                            slice(oz + iz_start, oz + iz_end + 1))

                            edgetask = EdgeTask(
                                    cgraph=cgraph,
                                    mysql_conn=mysql_conn,
                                    agglomeration_input=agglomeration_input,
                                    watershed_input=watershed_input,
                                    regiongraph_input=regiongraph_input,
                                    regiongraph_output=regiongraph_output,
                                    regiongraph_chunksize=regiongraph_chunksize,
                                    roi=edgetask_roi
                            )
                            edgetask.execute()

                            # Write relabeled ChunkGraph chunk to (possibly larger)
                            # watershed-chunk aligned buffer
                            watershed_output_buffer[ix_start:ix_end,
                                                    iy_start:iy_end,
                                                    iz_start:iz_end, :] = \
                                edgetask.get_relabeled_watershed()

                watershed_output[ox:ox + outer_chunksize[0],
                                 oy:oy + outer_chunksize[1],
                                 oz:oz + outer_chunksize[2], :] = \
                    watershed_output_buffer


if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    run_task_bundle(params, str_to_slice(sys.argv[2]))
