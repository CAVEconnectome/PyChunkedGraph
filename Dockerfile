FROM caveconnectome/pychunkedgraph:base_042124
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app