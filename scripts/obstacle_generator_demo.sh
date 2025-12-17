#!/bin/bash

set -ex

source /opt/ros/humble/local_setup.bash
pushd humble_ws
colcon build
popd
source humble_ws/install/local_setup.bash

for i in {1..10}
do
  ros2 launch obstacle_generator obstacle_generator.launch.py goal_text_file_path:=$PWD/builds/nav2/goals/goals_4.txt 
  sleep 0.5
done
