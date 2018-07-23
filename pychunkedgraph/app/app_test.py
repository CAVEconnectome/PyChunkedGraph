import time
import pytest
from pychunkedgraph.app import create_app


@pytest.fixture
def app():
    app = create_app(
        {
            'TESTING': True,
            'BIGTABLE_CONFIG': {
                'emulate': True
            }
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()

# TODO convert this to an actual self contained test with emulated backend
# and use app factory to create testing app and client objects

# TODO setup fixture that puts data in backend before running client tests


def request(test_client, op, body, post=True):

    if post:
        url = '/1.0/segment/{1}/{2}'.format(body[0], op)
        body = []
    else:
        url = '/1.0/graph/{1}'.format(op)

    print(url)
    time_start = time.time()
    response = test_client.get(url, verify=False, json=body)

    dt = (time.time() - time_start) * 1000
    print("%.3fms" % dt)

    return response


def get_root(client, atomic_id):
    body = [str(atomic_id), 0, 0, 0]

    print(body)
    r = request(client, "root", body, post=False)

    print(r.content)
    return r


def get_children(client, parent_id):
    body = [str(parent_id), 0, 0, 0]

    print(body)
    r = request(client, "children", body, post=True)

    # print(r.content)
    return r


def get_leaves(client, atomic_id):
    body = [str(atomic_id), 0, 0, 0]

    print(body)
    r = request(client, "leaves", body, post=True)

    # print(r.content)
    return r


def get_leaves_from_leave(atomic_id):
    body = [str(atomic_id), 0, 0, 0]

    print(body)
    r = request("leaves_from_leave", body, post=True)

    # print(r.content)
    return r


def merge(atomic_ids):
    body = [[str(atomic_ids[0]), 0, 0, 0],
            [str(atomic_ids[1]), 0, 0, 0]]

    print(body)
    r = request("merge", body, post=False)

    # print(r.content)
    return r


def split(atomic_ids):
    body = [[str(atomic_ids[0]), 0, 0, 0],
            [str(atomic_ids[1]), 0, 0, 0]]

    print(body)
    r = request("split", body, post=False)

    # print(r.content)
    return r
