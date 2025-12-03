#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /humble_ws/install/setup.bash

setsid ros2 launch carter_navigation carter_navigation_resim.launch.py "$@" &
LAUNCH_PID=$!

cleanup() {
    echo "Stopping ROS launch..."
    kill -TERM -$LAUNCH_PID 2>/dev/null || true
    wait $LAUNCH_PID 2>/dev/null || true

    mv /tmp/resim/outputs/record/* /tmp/resim/outputs 2>/dev/null || true
    rm -rf /tmp/resim/outputs/record 2>/dev/null || true
}
trap cleanup EXIT

while [ ! -f /tmp/isaac_ready ]; do
    # Check if launch already died while waiting
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        wait $LAUNCH_PID
        exit $?     # ← return launch exit code
    fi
    sleep 1
done

TIMEOUT="${RESIM_NAV2_TIMEOUT:-420}"
SECONDS=0
while [ $SECONDS -lt $TIMEOUT ]; do
    if ! kill -0 $LAUNCH_PID 2>/dev/null; then
        # Process ended early — get its exit code
        wait $LAUNCH_PID
        EXIT_CODE=$?
        echo "Scenario ended early. Exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
    sleep 1
done

echo "Timeout reached (${TIMEOUT}s), exiting."
echo $TIMEOUT > /tmp/resim/outputs/internal_timeout
exit 0
