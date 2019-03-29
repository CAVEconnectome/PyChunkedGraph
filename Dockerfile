FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY . /app
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /home/nginx/.cloudvolume/secrets \
  \
  && chown -R nginx /home/nginx \
  && usermod -d /home/nginx -s /bin/bash nginx \
  \
  && apt-get update \
  && apt-get install -y \
      # Boost and g++ for compiling igneous' C extensions (Mesher)
      build-essential \
      libboost-dev \
      # Required for adding graph-tools and cloud-sdk to the apt source list
      lsb-release \
      curl \
      apt-transport-https \
  \
  # GOOGLE-CLOUD-SDK
  && pip install --no-cache-dir --upgrade crcmod \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk-$(lsb_release -c -s) main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install -y google-cloud-sdk google-cloud-sdk-bigtable-emulator \
  \
  # PYTHON-GRAPH-TOOL
  # WARNING: This is ugly, graph-tools will use Debian's Python version and install as dist-util,
  #          but all our packages use the site-util Python version - we just create a sym_link,
  #          because it _seems_ to work and saves 80 minutes compilation time ...
  && echo "deb http://downloads.skewed.de/apt/$(lsb_release -s -c) $(lsb_release -s -c) main" > /etc/apt/sources.list.d/graph-tool.list \
  && echo "deb-src http://downloads.skewed.de/apt/$(lsb_release -s -c) $(lsb_release -s -c) main" >> /etc/apt/sources.list.d/graph-tool.list \
  && apt-key adv --no-tty --keyserver pgp.skewed.de --recv-key 612DEFB798507F25 \
  && apt-get update \
  && apt-get install -y python3-graph-tool \
  && ln -s /usr/lib/python3/dist-packages/graph_tool /usr/local/lib/python3.6/site-packages/graph_tool \
  && pip install --no-cache-dir --upgrade scipy \
  \
  # PYCHUNKEDGRAPH
  #   Need pip 18.1 for process-dependency-links flag support
  && pip install --no-cache-dir pip==18.1 \
  #   Need numpy to prevent install issue with cloud-volume / fpzip
  && pip install --no-cache-dir --upgrade numpy \
  && pip install --no-cache-dir --upgrade --process-dependency-links -e . \
  #   Tests
  && pip install tox codecov \
  && chmod +x tox_install_command.sh \
  \
  # CLEANUP
  #   libboost-dev and build-essentials will be required by tox to build python dependencies
  && apt-get remove --purge -y lsb-release curl \
  && apt-get autoremove --purge -y \
  && rm -rf /var/lib/apt/lists/* \
  && find /usr/local/lib/python3* -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' + \
  && find /usr/lib/python3* -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +
