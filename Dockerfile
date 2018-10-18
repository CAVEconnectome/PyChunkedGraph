FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN mkdir -p /home/nginx/.cloudvolume/secrets && chown -R nginx /home/nginx && usermod -d /home/nginx -s /bin/bash nginx
COPY requirements.txt /app/.
RUN pip install numpy &&  apt-get update \
  && apt-get install -y build-essential libboost-dev && git clone https://github.com/seung-lab/igneous.git && cd igneous \
  && python setup.py install && apt-get remove -y build-essential libboost-dev && rm -rf /var/lib/apt/lists/* && pip install -r requirements.txt
COPY . /app