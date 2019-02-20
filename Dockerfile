FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY . /app
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /home/nginx/.cloudvolume/secrets \
  \
  && chown -R nginx /home/nginx \
  && usermod -d /home/nginx -s /bin/bash nginx \
  \
  # Need boost and g++ for igneous meshing
  && apt-get update \
  && apt-get install -y build-essential libboost-dev \
  \
  # Need numpy to prevent install issue with cloud-volume/fpzip
  && pip install --no-cache-dir pip==18.1 \
  && pip install --no-cache-dir --upgrade numpy \
  # PyChunkedGraph
  && pip install --no-cache-dir --upgrade --process-dependency-links -e . \
  # Cleanup
  && apt-get remove -y build-essential libboost-dev \
  && rm -rf /var/lib/apt/lists/*
