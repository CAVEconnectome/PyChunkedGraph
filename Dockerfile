FROM caveconnectome/pychunkedgraph:base_042124
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY override/gcloud /app/venv/bin/gcloud
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# Hack to get zstandard from PyPI - remove if conda-forge linked lib issue is resolved
RUN  pip install --no-cache-dir --no-deps --force-reinstall zstandard==0.21.0
