FROM python:3-alpine

RUN apk add --no-cache --virtual .build-deps \
        curl \
        libc6-compat \
        git \
        gcc \
        g++ \
        linux-headers \
        jpeg-dev \
        mariadb-dev \
    && apk add --no-cache \
        libstdc++ \
        libjpeg-turbo \
        mariadb-connector-c \
    \
    # separate numpy install fixes cloudvolume bug
    && pip install --no-cache-dir \
        numpy \
    && pip install --no-cache-dir --upgrade \
        cloud-volume \
        tenacity \
        networkx \
        google-cloud-bigtable \
        zstandard \
        mysqlclient \
    && mkdir /root/.cloudvolume \
    && ln -s /secrets /root/.cloudvolume/secrets \
    \
    && git clone "https://github.com/seung-lab/pychunkedgraph.git" /usr/local/pychunkedgraph \
    && rm -rf /usr/local/pychunkedgraph/.git \
    && apk del .build-deps \
    && find /usr/local -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +
