from flask import Blueprint
from middle_auth_client import (
    auth_requires_admin,
    auth_required,
    auth_requires_permission,
)
from pychunkedgraph.app.segmentation import common

bp = Blueprint(
    "pcg_generic_v1", __name__, url_prefix=f"/{common.__segmentation_url_prefix__}"
)


# -------------------------------
# ------ Access control and index
# -------------------------------


@bp.route("/")
@bp.route("/index")
@auth_required
def index():
    return common.index()


@bp.route
@auth_required
def home():
    return common.home()


# -------------------
# ------ Applications
# -------------------


@bp.route("/sleep/<int:sleep>")
@auth_requires_admin
def sleep_me(sleep):
    return common.sleep_me(sleep)


@bp.route("/table/<table_id>/info", methods=["GET"])
@auth_requires_permission("view")
@common.remap_public
def handle_info(table_id):
    return common.handle_info(table_id)


# -------------------
# ------ API versions
# -------------------


@bp.route("/api/versions", methods=["GET"])
@auth_required
def handle_api_versions():
    return common.handle_api_versions()
