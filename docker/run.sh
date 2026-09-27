#!/usr/bin/env bash
# Run a command inside the ByteTrackV2 container with the repo mounted at /workspace.
# Usage: docker/run.sh [command ...]   (defaults to an interactive bash shell)
set -euo pipefail

IMAGE="${IMAGE:-bytetrack-v2:latest}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

if [ "$#" -eq 0 ]; then
    set -- bash
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
    --gpus all \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "${REPO_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE}" "$@"
