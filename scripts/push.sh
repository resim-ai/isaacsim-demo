#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ecr_compose_env.sh
source "${SCRIPT_DIR}/ecr_compose_env.sh"

docker push "${ISAACSIM_IMAGE}"
docker push "${NAV2_IMAGE}"
docker push "${METRICS_IMAGE}"
