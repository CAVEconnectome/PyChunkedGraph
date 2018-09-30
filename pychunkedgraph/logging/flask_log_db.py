import os
import json
from google.cloud import datastore

HOME = os.path.expanduser('~')

# Setting environment wide credential path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
           HOME + "/.cloudvolume/secrets/google-secret.json"

class FlaskLogDatabase(object):
    def __init__(self, table_id, project_id="neuromancer-seung-import",
                 client=None, credentials=None):
        self._table_id = table_id
        if client is not None:
            self._client = client
        else:
            self._client = datastore.Client(project=project_id,
                                            credentials=credentials)
    @property
    def table_id(self):
        return self._table_id

    @property
    def client(self):
        return self._client

    @property
    def namespace(self):
        return 'pychunkedgraphserverdb'

    @property
    def kind(self):
        return "flask_log_%s" % self.table_id

    def add_success_log(self, user_id, user_ip, request_time, response_time,
                        url, request_data=None):
        self._add_log(log_type="info", user_id=user_id, user_ip=user_ip,
                      request_time=request_time, response_time=response_time,
                      url=url, request_data=request_data)

    def add_internal_error_log(self, user_id, user_ip, request_time,
                               response_time, url, err_msg, request_data=None):
        self._add_log(log_type="internal_error", user_id=user_id,
                      user_ip=user_ip, request_time=request_time,
                      response_time=response_time, url=url,
                      request_data=request_data, msg=err_msg)

    def add_unhandled_exception_log(self, user_id, user_ip, request_time,
                                    response_time, url, err_msg,
                                    request_data=None):
        self._add_log(log_type="unhandled_exception", user_id=user_id,
                      user_ip=user_ip, request_time=request_time,
                      response_time=response_time, url=url,
                      request_data=request_data, msg=err_msg)

    def _add_log(self, log_type, user_id, user_ip, request_time, response_time,
                 url, request_arg=None, request_data=None, msg=None):
        # Extract relevant information and build entity

        key = self.client.key(self.kind, namespace=self.namespace)
        entity = datastore.Entity(key)

        url_split = url.split("/")

        if "?" in url_split[-1]:
            request_type = url_split[-1].split("?")[0]
            request_opt_arg = url_split[-1].split("?")[1]
        else:
            request_type = url_split[-1]
            request_opt_arg = None

        if len(request_data) == 0:
            request_data = None
        else:
            request_data = json.loads(request_data)

        entity['type'] = log_type
        entity['user_id'] = user_id
        entity['user_ip'] = user_ip
        entity['date'] = request_time
        entity['response_time(ms)'] = response_time
        entity['request_type'] = request_type
        entity['request_arg'] = request_arg
        entity['request_data'] = str(request_data)
        entity['request_opt_arg'] = request_opt_arg
        entity['url'] = url
        entity['msg'] = msg

        self.client.put(entity)

        return entity.key.id
