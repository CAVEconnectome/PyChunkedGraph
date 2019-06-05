from flask import Blueprint, request, make_response, jsonify, Response,\
    redirect, current_app
import json
import numpy as np


from pychunkedgraph.meshing import meshgen, meshgen_utils
from pychunkedgraph.app import app_utils
from pychunkedgraph.backend import chunkedgraph

# os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"

__version__ = '0.1.113'
bp = Blueprint('pychunkedgraph_meshing', __name__, url_prefix="/meshing")

# -------------------------------
# ------ Access control and index
# -------------------------------

@bp.route('/')
@bp.route("/index")
def index():
    return "Meshing Server -- " + __version__


@bp.route
def home():
    resp = make_response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    acah = "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers["Access-Control-Allow-Headers"] = acah
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Connection"] = "keep-alive"
    return resp


# ------------------------------------------------------------------------------

def _mesh_lvl2_nodes(serialized_cg_info, lvl2_nodes):
    cg = chunkedgraph.ChunkedGraph(**serialized_cg_info)

    for lvl2_node in lvl2_nodes:
        print(lvl2_node)
        meshgen.mesh_lvl2_preview(cg, lvl2_node, supervoxel_ids=None,
                                  cv_path=None, cv_mesh_dir=None, mip=2,
                                  simplification_factor=999999,
                                  max_err=40, parallel_download=1,
                                  verbose=True,
                                  cache_control='no-cache')

    return Response(status=200)



@bp.route('/1.0/<table_id>/<node_id>/mesh_preview', methods=['POST', 'GET'])
def handle_preview_meshes(table_id, node_id):
    if len(request.data) > 0:
        data = json.loads(request.data)
    else:
        data = {}

    node_id = np.uint64(node_id)

    cg = app_utils.get_cg(table_id)

    if "seg_ids" in data:
        seg_ids = data["seg_ids"]

        chunk_id = cg.get_chunk_id(node_id)
        supervoxel_ids = [cg.get_node_id(seg_id, chunk_id)
                          for seg_id in seg_ids]
    else:
        supervoxel_ids = None

    meshgen.mesh_lvl2_preview(cg, node_id, supervoxel_ids=supervoxel_ids,
                              cv_path=None, cv_mesh_dir=None, mip=2,
                              simplification_factor=999999,
                              max_err=40, parallel_download=1, verbose=True,
                              cache_control='no-cache')
    return Response(status=200)


## VALIDFRAGMENTS --------------------------------------------------------------

@bp.route('/1.0/<table_id>/<node_id>/validfragments', methods=['POST', 'GET'])
def handle_valid_frags(table_id, node_id):
    cg = app_utils.get_cg(table_id)

    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=1, verify_existence=True)

    return app_utils.tobinary(seg_ids)


## MANIFEST --------------------------------------------------------------------

@bp.route('/1.0/<table_id>/manifest/<node_id>:0', methods=['GET'])
def handle_get_manifest(table_id, node_id):
    if len(request.data) > 0:
        data = json.loads(request.data)
    else:
        data = {}

    if "start_layer" in data:
        start_layer = int(data["start_layer"])
    else:
        start_layer = None

    if "bounds" in request.args:
        bounds = request.args["bounds"]
        bounding_box = np.array([b.split("-") for b in bounds.split("_")],
                                dtype=np.int).T
    else:
        bounding_box = None

    verify = request.args.get('verify', False)
    verify = verify in ['True', 'true', '1', True]

    # TODO: Read this from config
    MESH_MIP = 2

    cg = app_utils.get_cg(table_id)
    seg_ids = meshgen_utils.get_highest_child_nodes_with_meshes(
        cg, np.uint64(node_id), stop_layer=2, start_layer=start_layer,
        bounding_box=bounding_box,
        verify_existence=verify)

    filenames = [meshgen_utils.get_mesh_name(cg, s, MESH_MIP) for s in seg_ids]

    return jsonify(fragments=filenames)
