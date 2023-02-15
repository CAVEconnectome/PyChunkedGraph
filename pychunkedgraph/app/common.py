# pylint: disable=invalid-name, missing-docstring, unspecified-encoding

import time
import traceback
from datetime import datetime

from cloudvolume import compression
from google.api_core.exceptions import GoogleAPIError
from flask import current_app, g, jsonify, request

from pychunkedgraph.logging.log_db import get_log_db


def before_request():
    current_app.request_start_time = time.time()
    current_app.request_start_date = datetime.utcnow()
    try:
        current_app.user_id = g.auth_user["id"]
    except (AttributeError, KeyError):
        current_app.user_id = "NA"
    current_app.table_id = None
    current_app.request_type = None
    content_encoding = request.headers.get("Content-Encoding", "")
    if "gzip" in content_encoding.lower():
        request.data = compression.decompress(request.data, "gzip")


def after_request(response):
    response_time = (time.time() - current_app.request_start_time) * 1000
    accept_encoding = request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept_encoding.lower():
        return response

    try:
        if current_app.table_id is not None:
            log_db = get_log_db(current_app.table_id)
            log_db.log_endpoint(
                path=request.full_path,
                user_id=current_app.user_id,
                request_ts=current_app.request_start_date,
                response_time=response_time,
            )
    except GoogleAPIError as e:
        current_app.logger.error(f"LogDB entry not successful: GoogleAPIError {e}")

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
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

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
    tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)

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
