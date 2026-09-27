#!/usr/bin/env bash
# Build the PETR/PETRv2 detector image. The build context is docker/petr only
# (everything is cloned inside the image), so datasets/ is never sent to the daemon.
set -euo pipefail
CTX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t "${IMAGE:-bytetrack-v2-petr:latest}" -f "${CTX_DIR}/Dockerfile" "${CTX_DIR}"
