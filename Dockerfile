# FROM gcr.io/neuromancer-seung-import/pychunkedgraph:graph-tool_dracopy
FROM caveconnectome/pychunkedgraph:base_042124
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY requirements.txt /app
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade -r requirements.txt
COPY . /app