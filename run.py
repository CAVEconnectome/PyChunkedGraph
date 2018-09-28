import sys
from werkzeug.serving import WSGIRequestHandler
import os

from pychunkedgraph.app import create_app

application = create_app()

if __name__ == '__main__':

    # Set HTTP protocol
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    # WSGIRequestHandler.protocol_version = "HTTP/2.0"

    application.run(host='0.0.0.0',
                    port=4000,
                    debug=True,
                    threaded=False,
                    ssl_context='adhoc')
