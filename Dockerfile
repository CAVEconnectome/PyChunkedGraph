# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=tiangolo/uwsgi-nginx-flask:python${PYTHON_VERSION}


######################################################
# Stage 1: Conda environment
######################################################
FROM ${BASE_IMAGE} AS conda-deps
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install build-essential wget -y \
  && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm Miniconda3-latest-Linux-x86_64.sh \
  && conda config --add channels conda-forge \
  && conda update -y --override-channels -c conda-forge conda \
  && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
  && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
  && conda install -y --override-channels -c conda-forge conda-pack

COPY requirements.yml requirements.txt requirements-dev.txt ./

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda env create -n pcg -f requirements.yml

RUN conda-pack -n pcg --ignore-missing-files -o /tmp/env.tar \
  && mkdir -p /app/venv && cd /app/venv \
  && tar xf /tmp/env.tar && rm /tmp/env.tar \
  && /app/venv/bin/conda-unpack


######################################################
# Stage 2: Bigtable emulator
######################################################
FROM golang:bullseye AS bigtable-emulator
ARG GOOGLE_CLOUD_GO_VERSION=bigtable/v1.19.0
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    git clone --depth=1 --branch="$GOOGLE_CLOUD_GO_VERSION" \
      https://github.com/googleapis/google-cloud-go.git /usr/src \
  && cd /usr/src/bigtable && go install -v ./cmd/emulator


######################################################
# Stage 3: Production
######################################################
FROM ${BASE_IMAGE}
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=conda-deps /app/venv /app/venv
COPY --from=bigtable-emulator /go/bin/emulator /app/venv/bin/cbtemulator
COPY override/gcloud /app/venv/bin/gcloud
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN pip install --no-cache-dir --no-deps --force-reinstall zstandard>=0.23.0 \
  && mkdir -p /home/nginx/.cloudvolume/secrets \
  && chown -R nginx /home/nginx \
  && usermod -d /home/nginx -s /bin/bash nginx

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade -r requirements.txt

COPY . /app
