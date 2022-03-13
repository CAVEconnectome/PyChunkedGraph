import sys
from werkzeug.serving import WSGIRequestHandler
import os

from pychunkedgraph import pcg_logger
from pychunkedgraph.app import create_app

app = create_app()

if __name__ == "__main__":

    assert len(sys.argv) >= 2
    HOME = os.path.expanduser("~")

    port = int(sys.argv[1])

    # Set HTTP protocol
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    # WSGIRequestHandler.protocol_version = "HTTP/2.0"

    pcg_logger.debug(app.config["PCG_GRAPH_IDS"])

    if len(sys.argv) == 2:
        app.run(
            host="0.0.0.0", port=port, debug=True, threaded=True, ssl_context="adhoc"
        )
    else:
        app.run(host="0.0.0.0", port=port, debug=True, threaded=True)

