#!/usr/bin/env bash
# Build the ByteTrackV2 image from the repo root context.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t "${IMAGE:-bytetrack-v2:latest}" -f "${REPO_DIR}/docker/Dockerfile" "${REPO_DIR}"
