#!/bin/bash

set -ex -o pipefail

export ROS_DISTRO=humble
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

export ISAACSIM_COMMAND="/isaac-sim/isaac-sim.streaming.sh"
if [[ -n "$GUI" ]]; then
  export ISAACSIM_COMMAND="/isaac-sim/isaac-sim.sh"
fi

exec $ISAACSIM_COMMAND --exec '/scripts/open_isaacsim_stage.py --path https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd --start-on-play' \
  --enable isaacsim.ros2.sim_control \
  --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge \
  --/renderer/shadercache/driverDiskCache/enabled=true \
  --/rtx/shaderDb/driverAppShaderCachePath=/shadercache \
  --/rtx/shaderDb/driverAppShaderCacheDirPerDriver=true \
  --/rtx/shaderDb/cachePermutationIndex=0
