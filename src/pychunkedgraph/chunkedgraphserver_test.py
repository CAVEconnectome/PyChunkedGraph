import requests
import time
import json

IP = "10.240.0.50"

def request(op, body, post=True):

    if post:
        url = 'https://{0}:4000/1.0/segment/{1}/{2}/'.format(IP, body[0], op)
        body = []
    else:
        url = 'https://{0}:4000/1.0/graph/{1}/'.format(IP, op)

    print(url)
    time_start = time.time()
    response = requests.get(url, verify=False, data=json.dumps(body))

    dt = (time.time() - time_start) * 1000
    print("%.3fms" % dt)

    return response


def get_root(atomic_id):
    body = [str(atomic_id), 0, 0, 0]

    print(body)
    r = request("root", body, post=False)

    print(r.content)
    return r


def get_children(parent_id):
    body = [str(parent_id), 0, 0, 0]

    print(body)
    r = request("children", body, post=True)

    # print(r.content)
    return r


def get_leaves(atomic_id):
    body = [str(atomic_id), 0, 0, 0]

    print(body)
    r = request("leaves", body, post=True)

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


