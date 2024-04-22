FROM caveconnectome/pychunkedgraph:base_042124
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY override/gcloud /app/venv/bin/gcloud
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt
COPY . /app
