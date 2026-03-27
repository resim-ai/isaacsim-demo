#!/bin/bash

set -eo pipefail

echo "Make sure you have launched Isaac Sim 5.1.0!"

# Set up directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR/.."

mkdir -p /tmp/resim

if [ ! -d /tmp/resim/assets ]; then
    ln -s "$PWD/assets" /tmp/resim/assets
fi

DEFAULT_EXPERIENCE_PATH="$PWD/builds/nav2/experiences/hospital/lobby_to_north_hallway_bright_no_ped.yaml"
EXPERIENCE_PATH="${1:-$DEFAULT_EXPERIENCE_PATH}"

if [ ! -f "$EXPERIENCE_PATH" ]; then
    echo "Experience file not found: $EXPERIENCE_PATH" >&2
    exit 1
fi

mkdir -p outputs
rm -rf outputs/*

if [ ! -d /tmp/resim/outputs ]; then
    ln -s "$PWD/outputs" /tmp/resim/outputs
fi

# Rebuild the workspace
$SCRIPT_DIR/rebuild_local.sh

source /opt/ros/humble/local_setup.bash
source humble_ws/install/local_setup.bash
source humble_ws/venv/bin/activate

ros2 launch carter_navigation carter_navigation_resim.launch.py \
    rviz:=true \
    send_goals:=true \
    experience_path:="$EXPERIENCE_PATH"

popd