#!/bin/bash

set -eo pipefail

source /opt/ros/humble/local_setup.bash

if [ ! -d humble_ws/venv/bin ]; then
    python3 -m venv --system-site-packages humble_ws/venv
    touch humble_ws/venv/COLCON_IGNORE
    source humble_ws/venv/bin/activate
    pip install -r humble_ws/requirements.txt
    deactivate
fi

pushd humble_ws
colcon build
popd
source humble_ws/install/local_setup.bash
