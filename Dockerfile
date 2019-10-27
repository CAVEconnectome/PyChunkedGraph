FROM gcr.io/neuromancer-seung-import/pychunkedgraph:base
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . /app
RUN pip install --no-cache-dir --upgrade --process-dependency-links -r requirements.txt