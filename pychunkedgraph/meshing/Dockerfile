FROM python:3-alpine
LABEL maintainer="Nico Kemnitz"

RUN apk add --no-cache --virtual .build-deps \
        curl \
        libc6-compat \
        git \
        gcc \
        g++ \
        linux-headers \
        jpeg-dev \
        zlib-dev \
        boost-dev \
    && apk add --no-cache \
        libstdc++ \
        libjpeg-turbo \
    \
    # separate numpy install fixes cloudvolume bug
    && pip install --no-cache-dir \
        numpy \
    && pip install --no-cache-dir --upgrade \
        cloud-volume \
        networkx \
        pandas \
        google-cloud-bigtable \
        google-api-python-client \
        git+https://github.com/sdorkenw/MultiWrapper.git#egg=multiwrapper \
    && pip install --no-cache-dir --upgrade --no-deps \
        task-queue \
    && mkdir /root/.cloudvolume \
    && ln -s /secrets /root/.cloudvolume/secrets \
    \
    && git clone "https://github.com/seung-lab/pychunkedgraph.git" /usr/local/pychunkedgraph \
    && cd /usr/local/pychunkedgraph \
    && python setup.py develop --upgrade --no-deps \
    && rm -rf /usr/local/pychunkedgraph/.git \
    && git clone "https://github.com/seung-lab/igneous.git" /usr/local/igneous \
    && cd /usr/local/igneous \
    && python setup.py develop --upgrade --no-deps \
    && rm -rf /usr/local/igneous/.git \
    && apk del .build-deps \
    && find /usr/local -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +
