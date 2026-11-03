#!/usr/bin/env bash
# build_and_run.sh
# Run this on the lab machine to build the image and (re)start fp-annotate.
# Usage:  bash build_and_run.sh
#
# The build pulls tensorflow-cpu (large) and can take a while — run this
# script inside a detached tmux session so a dropped SSH connection can't
# kill a mid-build download:
#   tmux new -s build && bash build_and_run.sh   # Ctrl-B D to detach
#   tmux attach -t build                          # to check back in later
#
# Mirrors the docker run invocation already proven to work: SDK at
# /opt/verifinger, unified /workspace, root (no --user).

set -euo pipefail

if [ -z "${TMUX:-}" ]; then
    echo "WARNING: not running inside tmux. This build pulls tensorflow-cpu and" >&2
    echo "can take a while — a dropped SSH connection will kill it. Consider" >&2
    echo "Ctrl-C and re-running as: tmux new -s build && bash $0" >&2
    sleep 3
fi

# ── Paths — adjust if your layout differs ─────────────────────────────────────
WORKSPACE_DIR="${HOME}/Desktop/synthetic_fingerprint_analysis/annotation_workspace"
VERIFINGER_SDK="${HOME}/Downloads/Neurotec_Biometric_2025_2_SDK_2026-04-03/Neurotec_Biometric_2025_2_SDK"
DATA_SYNTHETIC="/opt/data/FP_Synth"
DATA_REAL="/opt/data/FP_Real"

IMAGE_NAME="fingerprint-annotation:latest"
CONTAINER_NAME="fp-annotate"
DOCKERFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Build ──────────────────────────────────────────────────────────────────────
echo "==> Building image ${IMAGE_NAME} ..."
docker build --tag "${IMAGE_NAME}" "${DOCKERFILE_DIR}"

echo "==> Build complete."

# ── Remove stale container if it exists ───────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "==> Removing existing container '${CONTAINER_NAME}' ..."
    docker rm -f "${CONTAINER_NAME}"
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "==> Starting container '${CONTAINER_NAME}' ..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --hostname cenglab1 \
    -v /etc/machine-id:/etc/machine-id:ro \
    -v "${VERIFINGER_SDK}:/opt/verifinger:ro" \
    -v "${WORKSPACE_DIR}:/workspace" \
    -v "${DATA_SYNTHETIC}:/data/synthetic:ro" \
    -v "${DATA_REAL}:/data/real:ro" \
    "${IMAGE_NAME}"

echo ""
echo "==> Container '${CONTAINER_NAME}' is running (as root, no --user)."
echo "    Check readiness gate output: docker logs ${CONTAINER_NAME}"
echo "    Verify pgd is up:            docker exec ${CONTAINER_NAME} ps aux | grep '[.]/pgd'"
echo "    Re-run license smoke test:   docker exec ${CONTAINER_NAME} python3 /workspace/annotation_src/test_license.py"
