ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=tiangolo/uwsgi-nginx-flask:python${PYTHON_VERSION}


######################################################
# Build Image - PCG dependencies
######################################################
FROM ${BASE_IMAGE} AS pcg-build
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_ENV="pychunkedgraph"

# Setup Miniconda
RUN apt-get update && apt-get install build-essential wget -y
RUN wget \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && mkdir /root/.conda \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm -f Miniconda3-latest-Linux-x86_64.sh \
  && conda update conda

# Install PCG dependencies - especially graph-tool
# Note: uwsgi has trouble with pip and python3.11, so adding this with conda, too
COPY requirements.txt .
COPY requirements.yml .
COPY requirements-dev.txt .
RUN conda env create -n ${CONDA_ENV} -f requirements.yml

# Shrink conda environment into portable non-conda env
RUN conda install conda-pack -c conda-forge

RUN conda-pack -n ${CONDA_ENV} --ignore-missing-files -o /tmp/env.tar \
  && mkdir -p /app/venv \
  && cd /app/venv \
  && tar xf /tmp/env.tar \
  && rm /tmp/env.tar
RUN /app/venv/bin/conda-unpack


######################################################
# Build Image - Bigtable Emulator (without Google SDK)
######################################################
FROM golang:bullseye as bigtable-emulator-build
RUN mkdir -p /usr/src
WORKDIR /usr/src
ENV GOOGLE_CLOUD_GO_VERSION bigtable/v1.19.0
RUN apt-get update && apt-get install git -y
RUN git clone --depth=1 --branch="$GOOGLE_CLOUD_GO_VERSION" https://github.com/googleapis/google-cloud-go.git . \
  && cd bigtable \
  && go install -v ./cmd/emulator


######################################################
# Production Image
######################################################
FROM ${BASE_IMAGE}
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=pcg-build /app/venv /app/venv
COPY --from=bigtable-emulator-build /go/bin/emulator /app/venv/bin/cbtemulator
COPY override/gcloud /app/venv/bin/gcloud
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# Hack to get zstandard from PyPI - remove if conda-forge linked lib issue is resolved
RUN  pip install --no-cache-dir --no-deps --force-reinstall zstandard==0.21.0
RUN  pip install --no-cache-dir --no-deps --force-reinstall https://storage.googleapis.com/neuroglancer/ranl/protobuf/protobuf-4.24.2-cp311-abi3-linux_x86_64.whl
COPY . /app

RUN mkdir -p /home/nginx/.cloudvolume/secrets \
  && chown -R nginx /home/nginx \
  && usermod -d /home/nginx -s /bin/bash nginx
