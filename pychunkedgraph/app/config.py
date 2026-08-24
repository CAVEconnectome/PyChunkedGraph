# pylint: disable=invalid-name, missing-docstring, unspecified-encoding, line-too-long, too-few-public-methods

import logging
import os
import json
import datetime


class BaseConfig(object):
    DEBUG = False
    TESTING = False

    LOGGING_FORMAT = '{"source":"%(name)s","time":"%(asctime)s","severity":"%(levelname)s","message":"%(message)s"}'
    LOGGING_DATEFORMAT = "%Y-%m-%dT%H:%M:%S.0Z"
    LOGGING_LEVEL = logging.DEBUG

    # Opt-in start-of-request logging (see pychunkedgraph.app.common._log_request_start).
    # When True, every non-probe request emits a REQUEST_START line to stdout before any work
    # runs, so requests that OOM-kill their worker mid-flight (and thus never reach
    # after_request) are still visible in Cloud Logging. Verbose; enable only for temporary
    # diagnosis, e.g. by setting LOG_REQUEST_START = True in the instance config.cfg.
    LOG_REQUEST_START = False

    # Reject /lvl2_graph requests whose node resolves to more than this many level 2 nodes
    # (see pychunkedgraph.graph.analysis.pathing.get_lvl2_edge_list). Such objects — typically
    # erroneous mega-merges — produce a multi-GB induced edge list that can OOM the worker.
    # None disables the guard; set a concrete integer in the instance config.cfg to enable.
    LVL2_GRAPH_MAX_NODES = None

    # Guard for /subgraph (see pychunkedgraph.graph.subgraph.get_subgraph_edges_and_leaves).
    # Counts chunks rather than level 2 nodes: the endpoint reads every edge in every chunk the
    # object touches (all objects in the chunk, not just the requested one), so cost tracks the
    # volume queried, not the object. A large 'bounds' is expensive even for a small object.
    # None disables the guard; set a concrete integer in the instance config.cfg to enable.
    SUBGRAPH_MAX_CHUNKS = None

    CHUNKGRAPH_INSTANCE_ID = "pychunkedgraph"
    PROJECT_ID = os.environ.get("PROJECT_ID", None)
    CG_READ_ONLY = os.environ.get("CG_READ_ONLY", None) is not None
    PCG_GRAPH_IDS = os.environ.get("PCG_GRAPH_IDS", "").split(",")

    USE_REDIS_JOBS = False

    daf_credential_path = os.environ.get("DAF_CREDENTIALS", None)

    AUTH_TOKEN = None
    if daf_credential_path is not None:
        with open(daf_credential_path, "r") as f:
            AUTH_TOKEN = json.load(f)["token"]

    AUTH_SERVICE_NAMESPACE = "pychunkedgraph"
    VIRTUAL_TABLES = {
        "minnie65_public_v117": {
            "table_id": "minnie3_v1",
            "timestamp": datetime.datetime(
                year=2021,
                month=6,
                day=11,
                hour=8,
                minute=10,
                second=0,
                microsecond=253,
                tzinfo=datetime.timezone.utc,
            ),
        }
    }


class DevelopmentConfig(BaseConfig):
    """Development configuration."""

    USE_REDIS_JOBS = False
    DEBUG = True
    LOGGING_LEVEL = logging.ERROR


class DockerDevelopmentConfig(DevelopmentConfig):
    """Development configuration."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "dev")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class DeploymentWithRedisConfig(BaseConfig):
    """Deployment configuration with Redis."""

    USE_REDIS_JOBS = True
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"


class TestingConfig(BaseConfig):
    """Testing configuration."""

    TESTING = True
    USE_REDIS_JOBS = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
