# pylint: disable=invalid-name, missing-docstring

from flask import Blueprint
from middle_auth_client import (
    auth_required,
    auth_requires_admin,
    auth_requires_permission,
)

from pychunkedgraph.app import common as app_common
from pychunkedgraph.app.app_utils import remap_public
from pychunkedgraph.app.segmentation import common
from pychunkedgraph.graph import exceptions as cg_exceptions

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


# -------------------------------
# ------ Measurements and Logging
# -------------------------------


@bp.before_request
def before_request():
    return app_common.before_request()


@bp.after_request
def after_request(response):
    return app_common.after_request(response)


@bp.errorhandler(Exception)
def unhandled_exception(e):
    return app_common.unhandled_exception(e)


@bp.errorhandler(cg_exceptions.ChunkedGraphAPIError)
def api_exception(e):
    return app_common.api_exception(e)


# -------------------
# ------ Applications
# -------------------


@bp.route("/sleep/<int:sleep>")
@auth_requires_admin
def sleep_me(sleep):
    return common.sleep_me(sleep)


@bp.route("/table/<table_id>/info", methods=["GET"])
@auth_requires_permission("view", public_table_key="table_id")
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


@bp.route("/api/version", methods=["GET"])
@auth_required
def handle_version():
    return common.handle_version()
