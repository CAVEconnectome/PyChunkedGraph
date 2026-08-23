# pylint: disable=invalid-name, missing-docstring
import json
import os

import numpy as np
import redis
from rq import Queue, Connection, Retry
from flask import Response, current_app, g, jsonify, make_response, request

from pychunkedgraph import __version__
from pychunkedgraph.app import app_utils
from pychunkedgraph.app.meshing import tasks as meshing_tasks
from pychunkedgraph.meshing.manifest import get_highest_child_nodes_with_meshes
from pychunkedgraph.meshing.manifest import get_children_before_start_layer
from pychunkedgraph.meshing.manifest import ManifestCache


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

    cg = app_utils.get_cg(table_id)
    verify = request.args.get("verify", False)
    verify = verify in ["True", "true", "1", True]
    return_seg_ids = request.args.get("return_seg_ids", False)
    prepend_seg_ids = request.args.get("prepend_seg_ids", False)
    return_seg_ids = return_seg_ids in ["True", "true", "1", True]
    prepend_seg_ids = prepend_seg_ids in ["True", "true", "1", True]
    start_layer = cg.meta.custom_data.get("mesh", {}).get("max_layer", 2)
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
    )
    return manifest_response(cg, args)


def manifest_response(cg, args):
    from pychunkedgraph.meshing.manifest import speculative_manifest_sharded

    (
        node_id,
        verify,
        return_seg_ids,
        prepend_seg_ids,
        start_layer,
        flexible_start_layer,
        bounding_box,
        data,
    ) = args
    resp = {}
    seg_ids = []
    if not verify:
        seg_ids, resp["fragments"] = speculative_manifest_sharded(
            cg, node_id, start_layer=start_layer, bounding_box=bounding_box
        )

    else:
        seg_ids, resp["fragments"] = get_highest_child_nodes_with_meshes(
            cg,
            np.uint64(node_id),
            start_layer=start_layer,
            bounding_box=bounding_box,
        )
    if prepend_seg_ids:
        resp["fragments"] = [f"~{i}:{f}" for i, f in zip(seg_ids, resp["fragments"])]
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
def publish_remesh(table_id: str, user_id: str, lvl2_ids, is_priority: bool = True):
    """Enqueue a remesh onto the same Pub/Sub topic the edit path publishes to.

    Mirrors segmentation.common.publish_edit deliberately: one topic, and the
    `remesh_priority` attribute is what routes a message to a subscription. The
    infrastructure defines those subscriptions with attribute filters
    (terraform-google-cave/modules/local_cluster/pubsub.tf):

        <prefix>_PCG_HIGH_PRIORITY_REMESH   remesh_priority="true"   -> meshworker
        <prefix>_PCG_LOW_PRIORITY_REMESH    remesh_priority="false"  -> remeshworker

    so priority here is not a hint, it selects the consumer fleet.

    Note the same topic also feeds <prefix>_<ws>_L2CACHE_{HIGH,LOW}_PRIORITY_TRIGGER, so a
    manual remesh now also refreshes the l2 cache for these ids. That is intended -- a manual
    remesh usually follows a data problem, and the l2 cache derives from the same chunks -- but
    it is a real fan-out, not a no-op.
    """
    import pickle

    from messagingclient import MessagingClient

    attributes = {
        "table_id": table_id,
        "user_id": user_id,
        "remesh_priority": "true" if is_priority else "false",
        "remesh": "true",
    }
    payload = {
        # 0 means "no operation". A manual remesh has no GraphEditOperation behind it, and the
        # graph does not record which operation created a given level 2 node -- OperationID is
        # written only onto root-id rows (edits.py::_update_root_id_lineage), so the best available
        # answer is the latest operation on the whole object, which is not this node's provenance.
        # A wrong id in the worker's log line is worse than an honest unknown.
        #
        # The key must still exist and be int-convertible: mesh_worker.callback does
        # int(data["operation_id"]) unconditionally.
        "operation_id": 0,
        "new_lvl2_ids": np.asarray(lvl2_ids, dtype=np.uint64).tolist(),
        # Neither consumer reads these (mesh_worker uses new_lvl2_ids, the l2cache trigger uses
        # new_lvl2_ids); present so the payload shape stays identical to publish_edit's.
        "new_root_ids": [],
        "old_root_ids": [],
    }

    exchange = os.getenv("PYCHUNKEDGRAPH_EDITS_EXCHANGE", "pychunkedgraph")
    c = MessagingClient()
    c.publish(exchange, pickle.dumps(payload), attributes)


def handle_remesh(table_id):
    current_app.request_type = "remesh_enque"
    current_app.table_id = table_id
    # Same `priority` parameter, default, and semantics as every edit endpoint in
    # segmentation.common, so the two paths cannot drift. Unset means high priority, which is
    # the right default for an interactive request; a programmatic caller (caveclient, a
    # backfill script) should pass priority=false so bulk work lands on the low-priority
    # subscription and cannot starve human-triggered remeshes.
    is_priority = request.args.get("priority", True, type=str2bool)
    is_redisjob = request.args.get("use_redis", False, type=str2bool)

    new_lvl2_ids = json.loads(request.data)["new_lvl2_ids"]

    if is_redisjob:
        with Connection(redis.from_url(current_app.config["REDIS_URL"])):

            if is_priority:
                retry = Retry(max=3, interval=[1, 10, 60])
                queue_name = "mesh-chunks"
            else:
                retry = Retry(max=3, interval=[60, 60, 60])
                queue_name = "mesh-chunks-low-priority"
            q = Queue(queue_name, retry=retry, default_timeout=1200)
            task = q.enqueue(meshing_tasks.remeshing, table_id, new_lvl2_ids)

        response_object = {"status": "success", "data": {"task_id": task.get_id()}}

        return jsonify(response_object), 202
    else:
        # Publish, don't mesh here. This used to run meshgen.remeshing in a threading.Thread
        # inside the api pod, which put an unbounded, unretryable, invisible workload in a
        # request-serving process: the 202 was already returned, so a failure left no trace, and
        # a worker recycle, rollout or HPA scale-down silently discarded the work. Measured on
        # api6 2026-08-23, one remesh took the meshing pod from 198Mi to a 491Mi peak and left
        # it at 420Mi -- rss does not fall back -- so pods ratcheted up until they died: one had
        # reached 847Mi after 14h and was OOMKilled (whole cgroup, supervisord included) when a
        # remesh pushed it past its 1536Mi limit. The mesh workers exist for exactly this work,
        # request 3000Mi, and get retries and dead-lettering from Pub/Sub.
        new_lvl2_ids = np.array(new_lvl2_ids, dtype=np.uint64)

        if len(new_lvl2_ids) > 0:
            user_id = str(g.auth_user.get("id", current_app.user_id))
            publish_remesh(table_id, user_id, new_lvl2_ids, is_priority=is_priority)

        return Response(status=202)

    return Response(status=200)


def clear_manifest_cache(cg, node_id):
    node_ids = get_children_before_start_layer(cg, node_id, start_layer=2)
    ManifestCache(cg.graph_id).clear_fragments(node_ids)
