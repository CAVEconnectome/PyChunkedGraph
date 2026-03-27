# pylint: disable=invalid-name, missing-docstring, unspecified-encoding

import os
import json
import time
import traceback
from datetime import datetime, timezone

from cloudvolume import compression
from google.api_core.exceptions import GoogleAPIError
from flask import current_app, g, jsonify, request

from pychunkedgraph.logging.log_db import get_log_db

USER_NOT_FOUND = "-1"

ENABLE_LOGS = os.environ.get("PCG_SERVER_ENABLE_LOGS", "") != ""
LOG_LEAVES_MANY = os.environ.get("PCG_SERVER_LOGS_LEAVES_MANY", "") != ""


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
    current_app.request_start_date = datetime.now(timezone.utc)
    try:
        current_app.user_id = g.auth_user["id"]
    except (AttributeError, KeyError):
        current_app.user_id = USER_NOT_FOUND
    current_app.table_id = None
    current_app.operation_id = None
    current_app.request_type = None
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
