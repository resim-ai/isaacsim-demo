#!/bin/bash

set -euo pipefail

PROJECT_NAME="An Isaac Sim Sandbox"
REGISTRY="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images"

# Update SHA to the short commit SHA from the latest CI build (visible in ECR image tags).
SHA="b924075"

# Metrics build — update to the SHA tag produced by the latest CI run for the
# metrics container (tagged as isaac-sim-metrics-<sha> in ECR).
METRICS_IMAGE="${REGISTRY}:isaac-sim-metrics-${SHA}"
METRICS_VERSION="ee1de385c6e5eb71d41d27004b1931e3752aa91c"

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

# Set up the system build
export ISAACSIM_IMAGE="${REGISTRY}:isaacsim-mcb-isaacsim-${SHA}"
export NAV2_IMAGE="${REGISTRY}:isaacsim-mcb-nav2-${SHA}"
export ISAAC_SIM_BUILD_ID="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ ${SHA}" --description "Isaac Sim Nav2 demo build" \
  --branch "main" --version "${SHA}" --auto-create-branch --github --use-os-env \
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
