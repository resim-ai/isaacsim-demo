#!/bin/bash

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ecr_compose_env.sh
source "${SCRIPT_DIR}/ecr_compose_env.sh"

REGION="us-east-1"
ECR_REGISTRY="${ECR_REGISTRY_HOST}"

NAV2_LOCAL="resim-humble-nav2"
NAV2_REMOTE="${NAV2_IMAGE}"

METRICS_LOCAL="isaac-humble-metrics"
METRICS_REMOTE="${METRICS_IMAGE}"

ISAACSIM_LOCAL="resim-isaacsim"
ISAACSIM_REMOTE="${ISAACSIM_IMAGE}"

usage() {
    echo "Usage: $0 [--all] [--nav2] [--metrics] [--isaacsim]"
    echo "  --all       Build and push all images (default if no flags given)"
    echo "  --nav2      Build and push the nav2 image"
    echo "  --metrics   Build and push the metrics image"
    echo "  --isaacsim  Build and push the isaacsim image"
    echo "Tags match CI (.github/workflows/docker-build.yml). Short SHA from git HEAD,"
    echo "or set IMAGE_TAG_SHA to override (e.g. rebuild an existing tag)."
    exit 1
}

BUILD_NAV2=false
BUILD_METRICS=false
BUILD_ISAACSIM=false

if [ $# -eq 0 ]; then
    BUILD_NAV2=true
    BUILD_METRICS=true
    BUILD_ISAACSIM=true
fi

for arg in "$@"; do
    case $arg in
        --all)      BUILD_NAV2=true; BUILD_METRICS=true; BUILD_ISAACSIM=true ;;
        --nav2)     BUILD_NAV2=true ;;
        --metrics)  BUILD_METRICS=true ;;
        --isaacsim) BUILD_ISAACSIM=true ;;
        *) usage ;;
    esac
done

# Authenticate to ECR
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

if [ "${BUILD_ISAACSIM}" = true ]; then
    docker build . -f builds/isaacsim/isaacsim.dockerfile -t "${ISAACSIM_LOCAL}"
    docker tag "${ISAACSIM_LOCAL}" "${ISAACSIM_REMOTE}"
    docker push "${ISAACSIM_REMOTE}"
fi

if [ "${BUILD_NAV2}" = true ]; then
    docker build . -f builds/nav2/nav2.dockerfile -t "${NAV2_LOCAL}"
    docker tag "${NAV2_LOCAL}" "${NAV2_REMOTE}"
    docker push "${NAV2_REMOTE}"
fi

if [ "${BUILD_METRICS}" = true ]; then
    docker build . -f metrics/Dockerfile -t "${METRICS_LOCAL}"
    docker tag "${METRICS_LOCAL}" "${METRICS_REMOTE}"
    docker push "${METRICS_REMOTE}"
fi
