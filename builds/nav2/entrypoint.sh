#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /humble_ws/install/setup.bash

setsid ros2 launch carter_navigation carter_navigation_isaacsim.launch.py &
LAUNCH_PID=$!

cleanup() {
    echo "Stopping ROS launch..."
    kill -TERM -$LAUNCH_PID 2>/dev/null || true
    wait $LAUNCH_PID 2>/dev/null || true

    mv /tmp/resim/outputs/record/* /tmp/resim/outputs
    rm -rf /tmp/resim/outputs/record
}
trap cleanup EXIT

while [ ! -f /tmp/isaac_ready ]; do
    sleep 1
done

TIMEOUT="${RESIM_NAV2_TIMEOUT:-300}"
SECONDS=0
while [ $SECONDS -lt $TIMEOUT ]; do
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        echo "Scenario complete within time limit!"
        exit 0
    fi
    sleep 1
done

echo "Timeout reached (${TIMEOUT}s), exiting."
echo $TIMEOUT > /tmp/resim/outputs/internal_timeout
exit 0
