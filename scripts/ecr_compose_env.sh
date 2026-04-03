#!/usr/bin/env bash
# Shared ECR image URIs matching .github/workflows/docker-build.yml (docker/metadata-action tags).
# Source from repo root before docker compose or resim builds create --use-os-env, e.g.:
#   source scripts/ecr_compose_env.sh
# Override tag with: IMAGE_TAG_SHA=abc1234 source scripts/ecr_compose_env.sh

ECR_REGISTRY_HOST="${ECR_REGISTRY_HOST:-909785973729.dkr.ecr.us-east-1.amazonaws.com}"
ECR_REPO="${ECR_REPO:-isaacsim-test-images}"

# ECR tags use short SHA (same default as docker/metadata-action type=sha).
COMMIT_SHA="${IMAGE_TAG_SHA:-$(git rev-parse --short HEAD 2>/dev/null || true)}"
if [ -z "${COMMIT_SHA}" ]; then
	echo "ecr_compose_env.sh: set IMAGE_TAG_SHA or run from a git checkout" >&2
	return 1 2>/dev/null || exit 1
fi

# ReSim build --version and metrics-build --version use full SHA in CI (.github/workflows/docker-build.yml).
COMMIT_SHA_FULL="${COMMIT_SHA_FULL:-$(git rev-parse HEAD 2>/dev/null || true)}"

export COMMIT_SHA COMMIT_SHA_FULL ECR_REGISTRY_HOST ECR_REPO
export ISAACSIM_IMAGE="${ECR_REGISTRY_HOST}/${ECR_REPO}:isaacsim-mcb-isaacsim-${COMMIT_SHA}"
export NAV2_IMAGE="${ECR_REGISTRY_HOST}/${ECR_REPO}:isaacsim-mcb-nav2-${COMMIT_SHA}"
export METRICS_IMAGE="${ECR_REGISTRY_HOST}/${ECR_REPO}:isaac-sim-metrics-${COMMIT_SHA}"
