#!/bin/bash

set -ex -o pipefail

export ROS_DISTRO=humble
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="/fastdds.xml"

export ISAACSIM_COMMAND="/isaac-sim/isaac-sim.streaming.sh"
if [[ -n "$GUI" ]]; then
  export ISAACSIM_COMMAND="/isaac-sim/isaac-sim.sh"
fi

exec $ISAACSIM_COMMAND \
  --enable isaacsim.ros2.sim_control \
  --enable isaacsim.replicator.agent.core \
  --enable omni.anim.graph.core \
  --/app/scripting/ignoreWarningDialog=true \
  --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge \
  --/renderer/shadercache/driverDiskCache/enabled=true \
  --/rtx/shaderDb/driverAppShaderCachePath=/isaac-sim/shadercache \
  --/rtx/shaderDb/driverAppShaderCacheDirPerDriver=true \
  --/rtx/shaderDb/cachePermutationIndex=0
