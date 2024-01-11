# pylint: disable=invalid-name, missing-docstring, logging-fstring-interpolation

import os
from typing import Sequence
from time import mktime
from functools import wraps

import numpy as np
import networkx as nx
import requests
from flask import current_app, json, request
from scipy import spatial
from werkzeug.datastructures import ImmutableMultiDict

from pychunkedgraph import __version__
from pychunkedgraph.graph import ChunkedGraph
from pychunkedgraph.graph.client import get_default_client_info
from pychunkedgraph.graph import exceptions as cg_exceptions


PCG_CACHE = {}


def get_app_base_path():
    return os.path.dirname(os.path.realpath(__file__))


def get_instance_folder_path():
    return os.path.join(get_app_base_path(), "instance")


def remap_public(func=None, *, edit=False, check_node_ids=False):
    def mydecorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            virtual_tables = current_app.config.get("VIRTUAL_TABLES", None)

            # if not virtual configuration just return
            if virtual_tables is None:
                return f(*args, **kwargs)
            table_id = kwargs.get("table_id", None)
            http_args = request.args.to_dict()

            if table_id is None:
                # then no table remapping necessary
                return f(*args, **kwargs)
            if not table_id in virtual_tables:
                # if table table_id isn't in virtual
                # tables then just return
                return f(*args, **kwargs)
            else:
                # then we have a virtual table
                if edit:
                    raise cg_exceptions.Unauthorized(
                        "No edits allowed on virtual tables"
                    )
                # and we want to remap the table name
                new_table = virtual_tables[table_id]["table_id"]
                kwargs["table_id"] = new_table
                v_timestamp = virtual_tables[table_id]["timestamp"]
                v_timetamp_float = mktime(v_timestamp.timetuple())

                # we want to fix timestamp parameters too
                def ceiling_timestamp(argname):
                    old_arg = http_args.get(argname, None)
                    if old_arg is not None:
                        old_arg = float(old_arg)
                        # if they specified a timestamp
                        # enforce its less than the cap
                        if old_arg > v_timetamp_float:
                            http_args[argname] = v_timetamp_float
                    else:
                        # if they omit the timestamp, it defaults to "now"
                        # so we should cap it at the virtual timestamp
                        http_args[argname] = v_timetamp_float

                ceiling_timestamp("timestamp")
                ceiling_timestamp("timestamp_future")

                request.args = ImmutableMultiDict(http_args)

                # we also want to check for endpoints
                # which ask for info about IDs and
                # restrict such calls to IDs that are valid
                # before the timestamp cap for this virtual table
                cg = get_cg(new_table)

                def assert_node_prop(prop):
                    node_id = kwargs.get(prop, None)
                    if node_id is not None:
                        node_id = int(node_id)
                        # check if this root_id is valid at this timestamp
                        timestamp = cg.get_node_timestamps([node_id])
                        if not np.all(timestamp < np.datetime64(v_timestamp)):
                            raise cg_exceptions.Unauthorized(
                                "root_id not valid at timestamp"
                            )

                assert_node_prop("root_id")
                assert_node_prop("node_id")

                # some endpoints post node_ids as json, so we have to check there
                # as well if the endpoint configured us to.
                if check_node_ids:
                    node_ids = np.array(
                        json.loads(request.data)["node_ids"], dtype=np.uint64
                    )
                    timestamps = cg.get_node_timestamps(node_ids)
                    if not np.all(timestamps < np.datetime64(v_timestamp)):
                        raise cg_exceptions.Unauthorized(
                            "node_ids are all not valid at timestamp"
                        )

                return f(*args, **kwargs)

        return decorated_function

    if func:
        return mydecorator(func)
    else:
        return mydecorator


def jsonify_with_kwargs(data, as_response=True, **kwargs):
    kwargs.setdefault("separators", (",", ":"))

    if current_app.json.compact == False or current_app.debug:
        kwargs["indent"] = 2
        kwargs["separators"] = (", ", ": ")

    resp = json.dumps(data, **kwargs)
    if as_response:
        return current_app.response_class(
            resp + "\n", mimetype=current_app.json.mimetype
        )
    else:
        return resp


def ensure_correct_version(cg: ChunkedGraph) -> bool:
    current_major_version = int(__version__.split(".", maxsplit=1)[0])
    try:
        graph_major_version = int(cg.version.split(".")[0])
        valid = graph_major_version == current_major_version
        assert valid, f"v{cg.version} not supported, server version {__version__}."
        return True
    except (AttributeError, TypeError):
        # graph not versioned, later checked if whitelisted
        return False


def get_cg(table_id, skip_cache: bool = False):
    current_app.table_id = table_id
    if skip_cache is False:
        try:
            return PCG_CACHE[table_id]
        except KeyError:
            pass

    cg = ChunkedGraph(graph_id=table_id, client_info=get_default_client_info())
    version_valid = ensure_correct_version(cg)
    if version_valid:
        PCG_CACHE[table_id] = cg
        return cg

    if cg.graph_id in current_app.config["PCG_GRAPH_IDS"]:
        current_app.logger.warning(f"Serving whitelisted graph {cg.graph_id}.")
        PCG_CACHE[table_id] = cg
        return cg
    raise ValueError(f"Graph {cg.graph_id} not supported.")


def toboolean(value):
    """Transform value to boolean type.
    :param value: bool/int/str
    :return: bool
    :raises: ValueError, if value is not boolean.
    """
    if not value:
        raise ValueError("Can't convert null to boolean")

    if isinstance(value, bool):
        return value
    try:
        value = value.lower()
    except Exception as exc:
        raise ValueError(f"Can't convert {value} to boolean: {exc}") from exc

    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False

    raise ValueError(f"Can't convert {value} to boolean")


def tobinary(ids):
    """Transform id(s) to binary format

    :param ids: uint64 or list of uint64s
    :return: binary
    """
    return np.array(ids).tobytes()


def tobinary_multiples(arr):
    """Transform id(s) to binary format

    :param arr: list of uint64 or list of uint64s
    :return: binary
    """
    return [np.array(arr_i).tobytes() for arr_i in arr]


def handle_supervoxel_id_lookup(
    cg, coordinates: Sequence[Sequence[int]], node_ids: Sequence[np.uint64]
) -> Sequence[np.uint64]:
    """
    Helper to lookup supervoxel ids.
    This takes care of grouping coordinates.
    """

    def ccs(coordinates_nm_):
        graph = nx.Graph()
        dist_mat = spatial.distance.cdist(coordinates_nm_, coordinates_nm_)
        for edge in np.array(np.where(dist_mat < 1000)).T:
            graph.add_edge(*edge)
        ccs = [np.array(list(cc)) for cc in nx.connected_components(graph)]
        return ccs

    coordinates = np.array(coordinates, dtype=np.int)
    coordinates_nm = coordinates * cg.meta.resolution
    max_dist_steps = np.array([4, 8, 14, 28], dtype=float) * np.mean(cg.meta.resolution)

    node_ids = np.array(node_ids, dtype=np.uint64)
    if len(coordinates.shape) != 2:
        raise cg_exceptions.BadRequest(
            f"Could not determine supervoxel ID for coordinates "
            f"{coordinates} - Validation stage."
        )

    atomic_ids = np.zeros(len(coordinates), dtype=np.uint64)
    for node_id in np.unique(node_ids):
        node_id_m = node_ids == node_id
        for cc in ccs(coordinates_nm[node_id_m]):
            m_ids = np.where(node_id_m)[0][cc]

            for max_dist_nm in max_dist_steps:
                atomic_ids_sub = cg.get_atomic_ids_from_coords(
                    coordinates[m_ids], parent_id=node_id, max_dist_nm=max_dist_nm
                )
                if atomic_ids_sub is not None:
                    break
            if atomic_ids_sub is None:
                raise cg_exceptions.BadRequest(
                    f"Could not determine supervoxel ID for coordinates "
                    f"{coordinates} - Lookup stage."
                )
            atomic_ids[m_ids] = atomic_ids_sub
    return atomic_ids


def get_username_dict(user_ids, auth_token) -> dict:
    AUTH_URL = os.environ.get("AUTH_URL", None)
    if AUTH_URL is None:
        raise cg_exceptions.ChunkedGraphError("No AUTH_URL defined")

    users_request = requests.get(
        f"https://{AUTH_URL}/api/v1/username?id={','.join(map(str, np.unique(user_ids)))}",
        headers={"authorization": "Bearer " + auth_token},
        timeout=5,
    )
    return {x["id"]: x["name"] for x in users_request.json()}


def get_userinfo_dict(user_ids, auth_token):
    AUTH_URL = os.environ.get("AUTH_URL", None)

    if AUTH_URL is None:
        raise cg_exceptions.ChunkedGraphError("No AUTH_URL defined")

    users_request = requests.get(
        f"https://{AUTH_URL}/api/v1/user?id={','.join(map(str, np.unique(user_ids)))}",
        headers={"authorization": "Bearer " + auth_token},
        timeout=5,
    )
    return {x["id"]: x["name"] for x in users_request.json()}, {
        x["id"]: x["pi"] for x in users_request.json()
    }
