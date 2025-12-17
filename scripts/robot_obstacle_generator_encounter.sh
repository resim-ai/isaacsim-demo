#!/bin/bash

set -ex

source /opt/ros/humble/local_setup.bash
pushd humble_ws
colcon build
popd
source humble_ws/install/local_setup.bash

ros2 launch obstacle_generator obstacle_generator.launch.py goal_text_file_path:=$PWD/builds/nav2/goals/goals_4.txt seed:=4

ros2 service call /isaacsim/SetSimulationState simulation_interfaces/srv/SetSimulationState "{state: {state: 1}}"  # 1=playing

ros2 launch carter_navigation carter_navigation_resim.launch.py send_goals:=false rviz:=true &
nav2_pid=$!

# Cleanup function to kill nav2_pid on script exit
cleanup() {
    if [ -n "$nav2_pid" ] && kill -0 "$nav2_pid" 2>/dev/null; then
        kill "$nav2_pid"
    fi
    ros2 launch obstacle_generator cleanup_obstacles.launch.py
}

# Set up trap to call cleanup on script exit
trap cleanup EXIT INT TERM

sleep 20

ros2 launch isaac_ros_navigation_goal isaac_ros_navigation_goal.launch.py use_sim_time:=true goal_text_file_path:=$PWD/builds/nav2/goals/goals_4.txt


