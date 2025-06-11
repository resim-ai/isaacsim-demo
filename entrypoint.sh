#!/bin/bash

if  [ -f /tmp/resim/inputs/goals.txt ]; then
    cp /tmp/resim/inputs/goals.txt /humble_ws/src/navigation/isaac_ros_navigation_goal/assets/goals.txt
fi

source /opt/ros/humble/setup.bash
source /humble_ws/install/setup.bash
ros2 launch carter_navigation carter_navigation_isaacsim.launch.py

mv /tmp/resim/outputs/record/* /tmp/resim/outputs 
rm -rf /tmp/resim/outputs/record
