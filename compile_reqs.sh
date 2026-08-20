#!/bin/bash
# Recompile requirements.txt from requirements.in.
#
# The image only supplies the interpreter, so it has to match what the Dockerfile builds
# on (PYTHON_VERSION=3.11). The previous pin, v2.4.0, still ships Python 3.7, which no
# longer resolves the pinned dependency set -- caveclient alone requires >=3.11.
#
# --platform is explicit because these images are published for linux/amd64 only and the
# resolution has to target the deployment platform regardless of the build host.
#
# Writing to the existing requirements.txt is deliberate: pip-compile treats the current
# contents as constraints, so a run adds what is new instead of re-locking every pin. Pass
# -P/--upgrade-package or --upgrade explicitly when a bump is actually wanted.
set -euo pipefail

IMAGE="${PCG_COMPILE_IMAGE:-caveconnectome/pychunkedgraph:v2.22.0.dev8}"

docker run --rm --platform linux/amd64 -v "${PWD}":/app -w /app "${IMAGE}" /bin/bash -c "
  pip install --quiet pip-tools &&
  pip-compile requirements.in --resolver=backtracking --output-file requirements.txt
"
