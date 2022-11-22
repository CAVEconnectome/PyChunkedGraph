from flask import Blueprint
from middle_auth_client import (
    auth_requires_admin,
    auth_required,
    auth_requires_permission,
)
from pychunkedgraph.app.segmentation import common
from pychunkedgraph.app.app_utils import remap_public
import os

bp = Blueprint(
    "pcg_generic_v1", __name__, url_prefix=f"/{common.__segmentation_url_prefix__}"
)

if os.environ.get("DAF_CREDENTIALS", None) is not None:
    with open(os.environ.get("DAF_CREDENTIALS"), "r") as f:
        AUTH_TOKEN = json.load(f)["token"]
else:
    AUTH_TOKEN = ""

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
@auth_requires_permission("view", public_table_key='table_id', service_token=AUTH_TOKEN)
@remap_public
def handle_info(table_id):
    return common.handle_info(table_id)


# -------------------
# ------ API versions
# -------------------


@bp.route("/api/versions", methods=["GET"])
@auth_required
def handle_api_versions():
    return common.handle_api_versions()
