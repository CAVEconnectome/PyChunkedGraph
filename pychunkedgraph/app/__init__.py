from flask import Flask
from flask.logging import default_handler
from flask_cors import CORS
import sys
import logging
import os
import time
import json
import numpy as np
import datetime
from . import config
import redis
from rq import Queue

# from pychunkedgraph.app import app_blueprint
from pychunkedgraph.app import cg_app_blueprint, meshing_app_blueprint
from pychunkedgraph.logging import jsonformatter
# from pychunkedgraph.app import manifest_app_blueprint

from .redis_cli import init_redis_cmds
from ..ingest.cli import init_ingest_cmds

os.environ['TRAVIS_BRANCH'] = "IDONTKNOWWHYINEEDTHIS"


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

    app.register_blueprint(cg_app_blueprint.bp)
    app.register_blueprint(meshing_app_blueprint.bp)
    # app.register_blueprint(manifest_app_blueprint.bp)

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
        with app.app_context():
            init_ingest_cmds(app)
            init_redis_cmds(app)