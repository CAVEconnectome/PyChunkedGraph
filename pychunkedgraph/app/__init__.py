import datetime
import json
import logging
import os
import sys
import time

import pandas as pd
import numpy as np
import redis
from flask import Flask, request
from flask.json.provider import DefaultJSONProvider
from flask.logging import default_handler
from flask_cors import CORS
from rq import Queue

from pychunkedgraph import NOTICE, configure_logging
from pychunkedgraph.logging import jsonformatter

from . import config
from .meshing.legacy.routes import bp as meshing_api_legacy
from .meshing.v1.routes import bp as meshing_api_v1
from .segmentation.legacy.routes import bp as segmentation_api_legacy
from .segmentation.v1.routes import bp as segmentation_api_v1
from .segmentation.generic.routes import bp as generic_api
from .app_utils import get_instance_folder_path


class CustomJsonEncoder(json.JSONEncoder):
    def __init__(self, int64_as_str=False, **kwargs):
        super().__init__(**kwargs)
        self.int64_as_str = int64_as_str

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if self.int64_as_str and obj.dtype.type in (np.int64, np.uint64):
                return obj.astype(str).tolist()
            return obj.tolist()
        elif isinstance(obj, np.generic):
            if self.int64_as_str and obj.dtype.type in (np.int64, np.uint64):
                return obj.astype(str).item()
            return obj.item()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


class CustomJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(obj, default=None, cls=CustomJsonEncoder, **kwargs)


def create_app(test_config=None):
    app = Flask(
        __name__,
        instance_path=get_instance_folder_path(),
        instance_relative_config=True,
    )
    app.json = CustomJSONProvider(app)

    CORS(app, expose_headers="WWW-Authenticate")

    configure_app(app)

    if test_config is not None:
        app.config.update(test_config)

    app.register_blueprint(generic_api)

    app.register_blueprint(meshing_api_legacy)
    app.register_blueprint(meshing_api_v1)

    app.register_blueprint(segmentation_api_legacy)
    app.register_blueprint(segmentation_api_v1)

    _wire_post_edit_worker_recycle(app)

    return app


# Edit ops (split/merge) allocate multi-GB transient working sets that the
# allocator pools instead of returning to the OS, so RSS climbs across
# requests in long-lived workers. Recycling the worker after each successful
# edit caps RSS at the next-spawn baseline. uwsgi master logs one respawn
# line per recycle; no other noise.
_EDIT_ENDPOINTS = frozenset(
    {
        "pcg_segmentation_v1.handle_split",
        "pcg_segmentation_v1.handle_merge",
        "pcg_segmentation_v1.handle_merge_admin",
        "pcg_segmentation_v0.handle_split",
        "pcg_segmentation_v0.handle_merge",
    }
)


def _wire_post_edit_worker_recycle(app):
    try:
        import uwsgi  # type: ignore[import-not-found]
    except ImportError:
        return  # tests / dev server: no recycle

    @app.teardown_request
    def _recycle_worker(exc):
        if exc is not None or request.endpoint not in _EDIT_ENDPOINTS:
            return
        try:
            uwsgi.disconnect()
        except Exception:
            pass
        os._exit(0)


def configure_app(app):
    # Load logging scheme from config.py
    app_settings = os.getenv("APP_SETTINGS")
    if not app_settings:
        app.config.from_object(config.BaseConfig)
    else:
        app.config.from_object(app_settings)
    app.config.from_pyfile("config.cfg", silent=True)
    # Configure logging
    # handler = logging.FileHandler(app.config['LOGGING_LOCATION'])
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(app.config["LOGGING_LEVEL"])
    formatter = jsonformatter.JsonFormatter(
        fmt=app.config["LOGGING_FORMAT"], datefmt=app.config["LOGGING_DATEFORMAT"]
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    app.logger.removeHandler(default_handler)
    logging.getLogger().removeHandler(default_handler)
    app.logger.addHandler(handler)
    app.logger.setLevel(app.config["LOGGING_LEVEL"])
    app.logger.propagate = False

    # Ensure pychunkedgraph logger always works at NOTICE level
    # regardless of app config or environment log level
    configure_logging(level=NOTICE)
    pcg_logger = logging.getLogger("pychunkedgraph")
    # Root logger on the server image has a BASIC_FORMAT StreamHandler
    # (installed by uwsgi/gunicorn or an upstream basicConfig); propagating
    # past our own handler would re-emit every record in the
    # `LEVELNAME:logger.name:message` form.
    pcg_logger.propagate = False
    # app.logger.propagate = False blocks children under pychunkedgraph.app
    # from reaching the pychunkedgraph handler — attach it directly
    app_ns_logger = logging.getLogger("pychunkedgraph.app")
    for h in pcg_logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.NullHandler
        ):
            app_ns_logger.addHandler(h)
            break

    if app.config["USE_REDIS_JOBS"]:
        app.redis = redis.Redis.from_url(app.config["REDIS_URL"])
        app.test_q = Queue("test", connection=app.redis)
        with app.app_context():
            from ..ingest.rq_cli import init_rq_cmds
            from ..ingest.cli import init_ingest_cmds
            from ..ingest.cli_upgrade import init_upgrade_cmds

            init_rq_cmds(app)
            init_ingest_cmds(app)
            init_upgrade_cmds(app)
