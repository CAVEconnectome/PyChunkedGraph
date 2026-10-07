# pylint: disable=invalid-name, missing-docstring, unspecified-encoding

import os
import json
import time
import traceback
from datetime import datetime

from cloudvolume import compression
from google.api_core.exceptions import GoogleAPIError
from flask import current_app, g, jsonify, request

from pychunkedgraph.logging.log_db import get_log_db

USER_NOT_FOUND = "-1"

ENABLE_LOGS = os.environ.get("PCG_SERVER_ENABLE_LOGS", "") != ""
LOG_LEAVES_MANY = os.environ.get("PCG_SERVER_LOGS_LEAVES_MANY", "") != ""

# Health-check paths to skip, matched EXACTLY (not as prefixes) so the start-log signal
# isn't flooded by probes. These are the literal probe paths from the pychunkedgraph chart:
# the read/write deployments' readiness+liveness probes hit "/segmentation" and the GCP load
# balancer health check hits "/". Real API traffic lives under "/segmentation/api/..." and
# "/meshing/api/...", which are NOT equal to these entries and so are still logged.
_REQUEST_START_SKIP_PATHS = frozenset(("/", "/segmentation"))


def _log_request_start():
    # Emit a line to stdout at the *start* of a request, before any work runs, so it is
    # captured by Cloud Logging even if the request goes on to OOM-kill or otherwise crash
    # its worker before completing (such requests never reach after_request and so are
    # invisible in the Datastore server_logs completion logs). The `content_length` field is
    # the on-the-wire request body size (e.g. for a roots_binary POST, ~8 bytes per node id)
    # and `pid` is the uwsgi worker, so a spike/OOM can be traced to the specific in-flight
    # request and worker. Gated by the LOG_REQUEST_START Flask config value (default False);
    # verbose, intended for temporary diagnosis.
    try:
        user_id = g.auth_user["id"]
    except (AttributeError, KeyError):
        user_id = USER_NOT_FOUND
    current_app.logger.info(
        "REQUEST_START pid=%s method=%s path=%s content_length=%s user=%s remote=%s",
        os.getpid(),
        request.method,
        request.path,
        request.content_length,
        user_id,
        request.remote_addr,
    )


def _log_request(response_time):
    try:
        current_app.user_id = g.auth_user["id"]
    except (AttributeError, KeyError):
        current_app.user_id = USER_NOT_FOUND

    if ENABLE_LOGS is False:
        return

    if LOG_LEAVES_MANY is False and "leaves_many" in request.path:
        return

    try:
        if current_app.table_id is not None:
            log_db = get_log_db(current_app.table_id)
            args = dict(request.args)  # request.args is ImmutableMultiDict
            args.pop("middle_auth_token", None)
            log_db.log_endpoint(
                path=request.path,
                endpoint=request.endpoint,
                args=json.dumps(args),
                user_id=current_app.user_id,
                operation_id=current_app.operation_id,
                request_ts=current_app.request_start_date,
                response_time=response_time,
            )
    except GoogleAPIError as e:
        current_app.logger.error(f"LogDB entry not successful: GoogleAPIError {e}")


def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()
    try:
        current_app.user_id = g.auth_user["id"]
    except (AttributeError, KeyError):
        current_app.user_id = USER_NOT_FOUND
    current_app.table_id = None
    current_app.operation_id = None
    current_app.request_type = None
    if current_app.config.get("LOG_REQUEST_START", False) and (
        request.path not in _REQUEST_START_SKIP_PATHS
    ):
        _log_request_start()
    content_encoding = request.headers.get("Content-Encoding", "")
    if "gzip" in content_encoding.lower():
        request.data = compression.decompress(request.data, "gzip")


def after_request(response):
    response_time = (time.time() - current_app.request_start_time) * 1000
    accept_encoding = request.headers.get("Accept-Encoding", "")

    _log_request(response_time)

    if "gzip" not in accept_encoding.lower():
        return response

    response.direct_passthrough = False
    if (
        response.status_code < 200
        or response.status_code >= 300
        or "Content-Encoding" in response.headers
    ):
        return response

    response.data = compression.gzip_compress(response.data)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Content-Length"] = len(response.data)
    return response


def unhandled_exception(e):
    status_code = 500
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(e)

    _log_request(response_time)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": status_code,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": status_code,
        "message": str(e),
        "traceback": tb,
    }

    return jsonify(resp), status_code


def api_exception(e):
    response_time = (time.time() - current_app.request_start_time) * 1000
    user_ip = str(request.remote_addr)
    tb = traceback.format_exception(e)

    _log_request(response_time)

    current_app.logger.error(
        {
            "message": str(e),
            "user_id": user_ip,
            "user_ip": user_ip,
            "request_time": current_app.request_start_date,
            "request_url": request.url,
            "request_data": request.data,
            "response_time": response_time,
            "response_code": e.status_code.value,
            "traceback": tb,
        }
    )

    resp = {
        "timestamp": current_app.request_start_date,
        "duration": response_time,
        "code": e.status_code.value,
        "message": str(e),
    }
    return jsonify(resp), e.status_code.value
