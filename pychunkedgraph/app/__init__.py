import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import redis
from flask import Flask
from flask.logging import default_handler
from flask_cors import CORS
from rq import Queue

from pychunkedgraph.logging import jsonformatter

from . import config
from .meshing.legacy.routes import bp as meshing_api_legacy
from .meshing.v1.routes import bp as meshing_api_v1
from .segmentation.legacy.routes import bp as segmentation_api_legacy
from .segmentation.v1.routes import bp as segmentation_api_v1
from pychunkedgraph.app.segmentation.generic.routes import bp as generic_api


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)


def create_app(test_config=None):
    app = Flask(__name__)
    app.json_encoder = CustomJsonEncoder

    CORS(app, expose_headers='WWW-Authenticate')

    configure_app(app)

    if test_config is not None:
        app.config.update(test_config)

    app.register_blueprint(meshing_api_legacy)
    app.register_blueprint(meshing_api_v1)

    app.register_blueprint(segmentation_api_legacy)
    app.register_blueprint(segmentation_api_v1)

    app.register_blueprint(generic_api)
    return app


def configure_app(app):
    # Load logging scheme from config.py
    app_settings = os.getenv('APP_SETTINGS')
    if not app_settings:
        app.config.from_object(config.BaseConfig)
    else:
        app.config.from_object(app_settings)


    # Configure logging
    # handler = logging.FileHandler(app.config['LOGGING_LOCATION'])
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(app.config['LOGGING_LEVEL'])
    formatter = jsonformatter.JsonFormatter(
        fmt=app.config['LOGGING_FORMAT'],
        datefmt=app.config['LOGGING_DATEFORMAT'])
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    app.logger.removeHandler(default_handler)
    app.logger.addHandler(handler)
    app.logger.setLevel(app.config['LOGGING_LEVEL'])
    app.logger.propagate = False

    if app.config['USE_REDIS_JOBS']:
        app.redis = redis.Redis.from_url(app.config['REDIS_URL'])
        app.test_q = Queue('test', connection=app.redis)
