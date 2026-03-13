#!/bin/bash

set -euo pipefail

PROJECT_NAME="Isaac Sim Sandbox"

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
	--build-gpus 1 --build-memory-mib 16384 --build-vcpus 8 --metrics-build-vcpus 2 --metrics-build-memory-mib 8192
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
	--image 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaac-sim-metrics-ffe240b \
	--version "ffe240b" --github | sed 's/.*=//')"
# pause

# Set up the three system builds
# V1 - slow and a bit sad
export ISAACSIM_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-c6ff1a2"
export NAV2_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-c6ff1a2"
export ISAAC_SIM_BUILD_ID_V1="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ c6ff1a2" --description "c6ff1a2: [nav2] Add new metrics" \
  --branch "main" --version "c6ff1a2" --auto-create-branch --github --use-os-env | sed 's/.*=//')"
sleep 60
# pause

# V2 - new controller - faster
export ISAACSIM_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-ec89b0f"
export NAV2_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-ec89b0f"
export ISAAC_SIM_BUILD_ID_V2="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ ec89b0f" --description "ec89b0f: [nav2] Use MPPI controller" \
  --branch "main" --version "ec89b0f" --auto-create-branch --github --use-os-env | sed 's/.*=//')"
sleep 60
# pause

# V3 - new planner - smoother
export ISAACSIM_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-ffe240b"
export NAV2_IMAGE="909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-nav2-ffe240b"
export ISAAC_SIM_BUILD_ID_V3="$(resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
  --system "Isaac Sim" --name "Isaac Sim Build @ ffe240b" --description "ffe240b: [nav2] Use SmacLattice planner" \
  --branch "main" --version "ffe240b" --auto-create-branch --github --use-os-env | sed 's/.*=//')"
# pause

# Set up the test suite
resim test-suites create --project "${PROJECT_NAME}" \
	--name "Nav2 Demo Tests" --description "Tests for the Nav2 Demo" \
	--system "Isaac Sim" --metrics-build "${NAV2_METRICS_BUILD_ID}" \
	--show-on-summary \
	--experiences "Nav2 #1, Nav2 #2, Nav2 #3, Nav2 #4, Nav2 #5, Nav2 #6" \
  --metrics-set "Nav2 Metrics"
# pause

# Run the batches
resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 Baseline" \
  --build-id "${ISAAC_SIM_BUILD_ID_V1}" \
  --pool-labels 'resim:metrics2:k8s' \
  --sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml 
# pause

resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 New Controller" \
  --build-id "${ISAAC_SIM_BUILD_ID_V2}" \
  --pool-labels 'resim:metrics2:k8s' \
  --sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml 
# pause

resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "Nav2 Demo Tests" \
  --batch-name "Nav2 New Planner" \
  --build-id "${ISAAC_SIM_BUILD_ID_V3}" \
  --pool-labels 'resim:metrics2:k8s' \
  --sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml 
# pause
