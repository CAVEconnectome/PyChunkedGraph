# pylint: disable=invalid-name, missing-docstring

import datetime
import json
import logging
import os
import sys
import time

import pandas as pd
import numpy as np
import redis
from flask import Blueprint
from flask import Flask
from flask.json.provider import DefaultJSONProvider
from flask.logging import default_handler
from flask_cors import CORS
from rq import Queue

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

    auth_bp = Blueprint("auth_info", __name__, url_prefix="/")

    @auth_bp.route("/auth_info")
    def index():
        return {"login_url": "https://globalv1.flywire-daf.com/sticky_auth"}

    app.register_blueprint(auth_bp)

    return app


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
    app.logger.addHandler(handler)
    app.logger.setLevel(app.config["LOGGING_LEVEL"])
    app.logger.propagate = False

    if app.config["USE_REDIS_JOBS"]:
        app.redis = redis.Redis.from_url(app.config["REDIS_URL"])
        app.test_q = Queue("test", connection=app.redis)
        with app.app_context():
            from ..ingest.rq_cli import init_rq_cmds
            from ..ingest.cli import init_ingest_cmds

            init_rq_cmds(app)
            init_ingest_cmds(app)
