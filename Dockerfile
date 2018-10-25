FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN mkdir -p /home/nginx/.cloudvolume/secrets && chown -R nginx /home/nginx && usermod -d /home/nginx -s /bin/bash nginx
COPY requirements.txt /app/.
RUN pip install numpy \
  && pip install -r requirements.txt \
  && apt-get update \
  && apt-get install -y build-essential libboost-dev \
  && pip install --no-cache-dir --upgrade --no-deps google-api-python-client \
  && pip install --no-cache-dir --upgrade --no-deps task-queue \
  && git clone https://github.com/seung-lab/igneous.git \
  && cd igneous \
  && python setup.py develop --upgrade --no-deps \
  && apt-get remove -y build-essential libboost-dev \
  && rm -rf /var/lib/apt/lists/*
COPY timeout.conf /etc/nginx/conf.d/
COPY . /app
