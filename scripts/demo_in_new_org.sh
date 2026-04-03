#!/bin/bash

set -euo pipefail

PROJECT_NAME="Lain's Isaac Sim Sandbox"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ecr_compose_env.sh
source "${SCRIPT_DIR}/ecr_compose_env.sh"

# ReSim --version fields use full git SHA (same as CI).
if [ -z "${COMMIT_SHA_FULL}" ]; then
	echo "demo_in_new_org.sh: need a git checkout (or set COMMIT_SHA_FULL) for ReSim build/metrics versions" >&2
	exit 1
fi
METRICS_VERSION="${METRICS_VERSION:-${COMMIT_SHA_FULL}}"

# Function to pause execution
pause() {
    echo "Press Enter to continue to the next command..."
    read -p ""
}

echo "Have you changed your current org & do you have yq installed?"
read -p "Press Enter to continue"

# Create a new project and system
resim projects create --name "${PROJECT_NAME}" --description "A project for running the Nav2 Demo"
# pause

resim system create --project "${PROJECT_NAME}" --name "Isaac Sim" --description "A system for running Isaac Sim" \
	--build-gpus 1 --build-memory-mib 32768 --build-vcpus 8 --metrics-build-vcpus 2 --metrics-build-memory-mib 8192
# pause

# Sync all experiences and managed test suites from resim_experience_sync.yaml.
# This creates/updates all 46 hospital + warehouse experiences and the three
# managed test suites (Hospital Demo, Demo Smoke, Warehouse Demo).
resim experiences sync --project "${PROJECT_NAME}" --file ./resim_experience_sync.yaml
# pause

# Set up the metrics builds
export NAV2_METRICS_BUILD_ID="$(resim metrics-builds create --project "${PROJECT_NAME}" \
	--name "Nav2 Metrics" \
	--image "${METRICS_IMAGE}" \
	--version "${METRICS_VERSION}" \
	--systems "Isaac Sim" \
	--github | sed 's/.*=//')"
# pause

export DEFAULT_REPORT_METRICS_BUILD_ID="$(resim metrics-builds create --project "${PROJECT_NAME}" \
	--name "Default Report Metrics Build" \
	--image public.ecr.aws/resim/open-metrics-builds/default-reports-build:sha-b64ef17bcd45d62bf0c1a09cad168372ffe87e9c \
	--version "b64ef17bcd45d62bf0c1a09cad168372ffe87e9c" \
	--github | sed 's/.*=//')"
# pause

# Set up the system build (ISAACSIM_IMAGE / NAV2_IMAGE / METRICS_IMAGE from ecr_compose_env.sh)
export ISAAC_SIM_BUILD_ID="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ ${COMMIT_SHA}" --description "Isaac Sim Nav2 demo build" \
  --branch "main" --version "${COMMIT_SHA_FULL}" --auto-create-branch --github --use-os-env \
  --assets "collected_hospital_demo:0,carter_warehouse_navigation_collected:0" | sed 's/.*=//')"
# pause

# Attach the metrics build and metrics set to the Demo Smoke suite before running
resim suites revise --project "${PROJECT_NAME}" \
  --test-suite "Demo Smoke" \
  --metrics-build "${NAV2_METRICS_BUILD_ID}" \
  --metrics-set "Nav2 Metrics"
# pause

# Run the batch
resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Demo Smoke" \
  --batch-name "Nav2 Demo" \
  --build-id "${ISAAC_SIM_BUILD_ID}" \
  --pool-labels 'resim:metrics2:k8s' \
  --sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml \
  --allowable-failure-percent 25
# pause

echo "Once the above batch is complete, run the report using the following command: "
echo "resim reports create --branch main --metrics-build-id $DEFAULT_REPORT_METRICS_BUILD_ID --project \"${PROJECT_NAME}\" --test-suite \"Demo Smoke\" --length 1"
