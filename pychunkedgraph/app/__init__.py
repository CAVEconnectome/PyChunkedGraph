from flask import Flask
from flask_cors import CORS
import sys
import logging

from . import config

# from pychunkedgraph.app import app_blueprint
from pychunkedgraph.app import cg_app_blueprint
# from pychunkedgraph.app import manifest_app_blueprint


def create_app(test_config=None):
    app = Flask(__name__)
    CORS(app)

    configure_app(app)

    if test_config is not None:
        app.config.update(test_config)

    app.register_blueprint(cg_app_blueprint.bp)
    # app.register_blueprint(manifest_app_blueprint.bp)

    return app


def configure_app(app):

    # Load logging scheme from config.py
    app.config.from_object(config.BaseConfig)

    # Configure logging
    # handler = logging.FileHandler(app.config['LOGGING_LOCATION'])
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(app.config['LOGGING_LEVEL'])
    formatter = logging.Formatter(app.config['LOGGING_FORMAT'])
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)


