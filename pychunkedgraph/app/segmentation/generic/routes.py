from flask import Blueprint

from pychunkedgraph.app.segmentation import common

bp = Blueprint("pcg_generic_v1", __name__, url_prefix="/segmentation")


# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route("/")
@bp.route("/index")
def index():
    return common.index()


@bp.route
def home():
    return common.home()


# -------------------
# ------ Applications
# -------------------


@bp.route("/sleep/<int:sleep>")
def sleep_me(sleep):
    return common.sleep_me(sleep)


@bp.route("/table/<table_id>/info", methods=["GET"])
def handle_info(table_id):
    return common.handle_info(table_id)


# -------------------
# ------ API versions
# -------------------


@bp.route("/api/versions", methods=["GET"])
def handle_info(table_id):
    return common.handle_api_versions()