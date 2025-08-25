#!/bin/bash

set -euo pipefail

PROJECT_NAME="Isaac Sim x ReSim"

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

# # Setup the experiences
EXPERIENCE_NUMBERS=(1 2 3 4 5 6)
for EXPERIENCE_NUMBER in "${EXPERIENCE_NUMBERS[@]}"; do
	resim experiences create --project "${PROJECT_NAME}" \
		--name "Nav2 #${EXPERIENCE_NUMBER}" \
		--description "Nav2 Experience ${EXPERIENCE_NUMBER}" \
		--location "/goals/goals_${EXPERIENCE_NUMBER}.txt" \
    --systems "Isaac Sim"
	# pause
done

# # Set up the metrics builds
export NAV2_METRICS_BUILD_ID="$(resim metrics-builds create --project "${PROJECT_NAME}" \
	--name "Nav2 Metrics" \
	--image 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaac-sim-metrics-19995ae \
	--version "ee1de385c6e5eb71d41d27004b1931e3752aa91c" --github | sed 's/.*=//')"
# pause

export DEFAULT_REPORT_METRICS_BUILD_ID="$(resim metrics-builds create --project "${PROJECT_NAME}" \
	--name "Default Report Metrics Build" \
	--image public.ecr.aws/resim/open-metrics-builds/default-reports-build:sha-b64ef17bcd45d62bf0c1a09cad168372ffe87e9c \
	--version "b64ef17bcd45d62bf0c1a09cad168372ffe87e9c" \
	--github | sed 's/.*=//')"
# pause

# Set up the test suite
resim test-suites create --project "${PROJECT_NAME}" \
	--name "Nav2 Demo Tests" --description "Tests for the Nav2 Demo" \
	--system "Isaac Sim" --metrics-build "${NAV2_METRICS_BUILD_ID}" \
	--show-on-summary \
	--experiences "Nav2 #1, Nav2 #2, Nav2 #3, Nav2 #4, Nav2 #5, Nav2 #6"
# pause

# Set up the three system builds
# V1 - slow and a bit sad
yq -iy '.services.isaacsim.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-bdbf838"' ./builds/docker-compose.yml
yq -iy '.services.nav2.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-bdbf838"' ./builds/docker-compose.yml
export ISAAC_SIM_BUILD_ID_V1="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ bdbf838" --description "bdbf838: [nav2] Add new metrics" \
  --branch "main" --version "bdbf838" --auto-create-branch --github | sed 's/.*=//')"
sleep 60
# pause

# V2 - new controller - faster
yq -iy '.services.isaacsim.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-33edb5e"' ./builds/docker-compose.yml
yq -iy '.services.nav2.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-33edb5e"' ./builds/docker-compose.yml
export ISAAC_SIM_BUILD_ID_V2="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ 33edb5e" --description "33edb5e: [nav2] Use MPPI controller" \
  --branch "main" --version "33edb5e" --auto-create-branch --github | sed 's/.*=//')"
sleep 60
# pause

# V3 - new planner - smoother
yq -iy '.services.isaacsim.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-19995ae"' ./builds/docker-compose.yml
yq -iy '.services.nav2.image = "909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-19995ae"' ./builds/docker-compose.yml
export ISAAC_SIM_BUILD_ID_V3="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ 19995ae" --description "19995ae: [nav2] Use SmacLattice planner" \
  --branch "main" --version "19995ae" --auto-create-branch --github | sed 's/.*=//')"
sleep 60
# pause

# Run the batches
resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 Baseline" \
  --build-id "${ISAAC_SIM_BUILD_ID_V1}"
# pause


resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 New Controller" \
  --build-id "${ISAAC_SIM_BUILD_ID_V2}"
# pause

resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 New Planner" \
  --build-id "${ISAAC_SIM_BUILD_ID_V3}"
# pause

echo "Once the above batches are complete, run the report using the following command: "
echo "resim reports create --branch main --metrics-build-id $DEFAULT_REPORT_METRICS_BUILD_ID --project "${PROJECT_NAME}" --test-suite \"Nav2 Demo Tests\" --length 1"