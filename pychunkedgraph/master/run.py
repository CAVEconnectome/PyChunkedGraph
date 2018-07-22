from pychunkedgraph.master import create_app
import sys
from werkzeug.serving import WSGIRequestHandler
import os
app = create_app()

if __name__ == '__main__':

    assert len(sys.argv) == 3
    HOME = os.path.expanduser("~")

    table_id = sys.argv[1]
    port = int(sys.argv[2])
    app.config['table_id'] = table_id
    # Initialize chunkedgraph:
    # cg = chunkedgraph.ChunkedGraph(table_id=table_id)

    # Initialize google pubsub publisher
    # publisher = pubsub_v1.PublisherClient()
    # topic_path = publisher.topic_path('neuromancer-seung-import',
    #                                   'pychunkedgraph')

    # Set HTTP protocol
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    # WSGIRequestHandler.protocol_version = "HTTP/2.0"

    print("Table: %s; Port: %d; Log-Path: %s" %
          (table_id, port, app.config['LOGGING_LOCATION']))

    app.run(host='0.0.0.0',
            port=port,
            debug=True,
            threaded=True,
            ssl_context=(HOME + '/keys/server.crt',
                         HOME + '/keys/server.key'))
