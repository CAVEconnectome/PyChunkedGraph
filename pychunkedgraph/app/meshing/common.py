# pylint: disable=invalid-name, missing-docstring
import json
import os
import threading

import numpy as np
from flask import Response, current_app, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.graph import chunkedgraph
from pychunkedgraph.graph import exceptions as cg_exceptions

__meshing_url_prefix__ = os.environ.get("MESHING_URL_PREFIX", "meshing")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def index():
    return f"PyChunkedGraph Meshing v{__version__}"


def home():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


## VALIDFRAGMENTS --------------------------------------------------------------


def handle_valid_frags(table_id, node_id):
    # nested: pulls meshing/cloudvolume, only needed at call time
    from pychunkedgraph.meshing.manifest import get_highest_child_nodes_with_meshes

    current_app.table_id = table_id
    cg = app_utils.get_cg(table_id)
    seg_ids = get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True
    )
    return app_utils.tobinary(seg_ids)


## MANIFEST --------------------------------------------------------------------


def handle_get_manifest(table_id, node_id):
    current_app.request_type = "manifest"
    current_app.table_id = table_id

    data = {}
    if len(request.data) > 0:
        data = json.loads(request.data)

    bounding_box = None
    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")], dtype=int).T

    from pychunkedgraph.meshing.mesh_meta import MeshMeta
    from pychunkedgraph.meshing.manifest import v2

    cg = app_utils.get_cg(table_id)
    mm = MeshMeta(cg)
    manifest_version = v2.requested_manifest_version(request.headers.get("Accept"))
    if manifest_version < 2 and mm.needs_v2:
        raise cg_exceptions.NotAcceptable(
            "This dataset serves meshes from an absolute path; upgrade your "
            "client to one that requests "
            "'Accept: application/x.cave;manifest_version=2'."
        )
    verify = request.args.get("verify", False)
    verify = verify in ["True", "true", "1", True]
    return_seg_ids = request.args.get("return_seg_ids", False)
    prepend_seg_ids = request.args.get("prepend_seg_ids", False)
    return_seg_ids = return_seg_ids in ["True", "true", "1", True]
    prepend_seg_ids = prepend_seg_ids in ["True", "true", "1", True]
    start_layer = mm.max_layer
    start_layer = int(request.args.get("start_layer", start_layer))
    if "start_layer" in data:
        start_layer = int(data["start_layer"])

    flexible_start_layer = None
    if "flexible_start_layer" in data:
        flexible_start_layer = int(data["flexible_start_layer"])
    args = (
        node_id,
        verify,
        return_seg_ids,
        prepend_seg_ids,
        start_layer,
        flexible_start_layer,
        bounding_box,
        data,
        manifest_version,
    )
    response = make_response(jsonify(manifest_response(cg, args)))
    response.headers["X-Manifest-Version"] = str(manifest_version)
    response.headers["Vary"] = "Accept"
    return response


def manifest_response(cg, args):
    from pychunkedgraph.meshing.manifest import speculative_manifest_sharded
    from pychunkedgraph.meshing.manifest import get_highest_child_nodes_with_meshes
    from pychunkedgraph.meshing.manifest import v2
    from pychunkedgraph.meshing.mesh_meta import MeshMeta

    (
        node_id,
        verify,
        return_seg_ids,
        prepend_seg_ids,
        start_layer,
        flexible_start_layer,
        bounding_box,
        data,
        manifest_version,
    ) = args
    if not verify:
        seg_ids, fragments = speculative_manifest_sharded(
            cg, node_id, start_layer=start_layer, bounding_box=bounding_box
        )
    else:
        seg_ids, fragments = get_highest_child_nodes_with_meshes(
            cg,
            np.uint64(node_id),
            start_layer=start_layer,
            bounding_box=bounding_box,
        )

    if manifest_version >= 2:
        mm = MeshMeta(cg)
        initial, dynamic = v2.to_v2_groups(
            seg_ids, fragments, seg_id_in_fragment=not verify
        )
        resp = v2.assemble(mm.initial_path, mm.dynamic_path, initial, dynamic)
    else:
        resp = {"fragments": fragments}
        if prepend_seg_ids:
            resp["fragments"] = [f"~{i}:{f}" for i, f in zip(seg_ids, fragments)]
        if return_seg_ids:
            resp["seg_ids"] = seg_ids
    return _check_post_options(cg, resp, data, seg_ids)


def _check_post_options(cg, resp, data, seg_ids):
    if app_utils.toboolean(data.get("return_seg_ids", "false")):
        resp["seg_ids"] = seg_ids
    if app_utils.toboolean(data.get("return_seg_id_layers", "false")):
        resp["seg_id_layers"] = cg.get_chunk_layers(seg_ids)
    if app_utils.toboolean(data.get("return_seg_chunk_coordinates", "false")):
        resp["seg_chunk_coordinates"] = [
            cg.get_chunk_coordinates(seg_id) for seg_id in seg_ids
        ]
    return resp


## REMESHING -----------------------------------------------------
def handle_remesh(table_id):
    current_app.request_type = "remesh_enque"
    current_app.table_id = table_id
    new_lvl2_ids = json.loads(request.data)["new_lvl2_ids"]
    new_lvl2_ids = np.array(new_lvl2_ids, dtype=np.uint64)
    cg = app_utils.get_cg(table_id)
    if len(new_lvl2_ids) > 0:
        t = threading.Thread(
            target=_remeshing, args=(cg.get_serialized_info(), new_lvl2_ids)
        )
        t.start()
    return Response(status=202)


def _remeshing(serialized_cg_info, lvl2_nodes):
    # nested: pulls meshing/cloudvolume, only needed at call time
    from pychunkedgraph.meshing import meshgen
    from pychunkedgraph.meshing.mesh_meta import MeshMeta

    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)
    mm = MeshMeta(cg)
    meshgen.remeshing(
        cg,
        lvl2_nodes,
        stop_layer=mm.max_layer,
        mip=mm.mip,
        max_err=mm.max_error,
        cv_sharded_mesh_dir=mm.dir,
        cv_unsharded_mesh_path=mm.dynamic_path,
    )

    return Response(status=200)


def clear_manifest_cache(cg, node_id):
    # nested: pulls meshing/cloudvolume, only needed at call time
    from pychunkedgraph.meshing.manifest import get_children_before_start_layer
    from pychunkedgraph.meshing.manifest import ManifestCache

    node_ids = get_children_before_start_layer(cg, node_id, start_layer=2)
    ManifestCache(cg.graph_id).clear_fragments(node_ids)


def clear_manifest_cache_all(cg) -> int:
    """Delete every cached manifest fragment for this graph.

    Returns the number of redis keys deleted across both initial and
    dynamic caches (they share the ``<graph_id>:`` namespace).
    """
    # nested: pulls meshing/cloudvolume, only needed at call time
    from pychunkedgraph.meshing.manifest import ManifestCache

    return ManifestCache(cg.graph_id).clear_namespace()
