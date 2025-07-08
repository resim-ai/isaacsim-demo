#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /humble_ws/install/setup.bash
ros2 launch carter_navigation carter_navigation_isaacsim.launch.py

mv /tmp/resim/outputs/record/* /tmp/resim/outputs 
rm -rf /tmp/resim/outputs/record
