import sys
from werkzeug.serving import WSGIRequestHandler

from pychunkedgraph.app import create_app

app = create_app()

if __name__ == "__main__":
    assert len(sys.argv) >= 2

    port = int(sys.argv[1])
    WSGIRequestHandler.protocol_version = "HTTP/1.1"

    if len(sys.argv) == 2:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=True,
            threaded=True,
            ssl_context="adhoc",
        )
    else:
        app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
