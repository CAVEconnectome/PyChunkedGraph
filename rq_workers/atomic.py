import os
# This is for monitoring rq with supervisord
# For the flask app use a config class

from pychunkedgraph.utils.redis import REDIS_URL

# Queues to listen on
QUEUES = ["atomic"]

# If you're using Sentry to collect your runtime exceptions, you can use this
# to configure RQ for it in a single step
# The 'sync+' prefix is required for raven: https://github.com/nvie/rq/issues/350#issuecomment-43592410
# SENTRY_DSN = 'sync+http://public:secret@example.com/1'

# If you want custom worker name
# NAME = 'worker-1024'
