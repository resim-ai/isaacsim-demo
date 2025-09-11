#!/bin/bash
source /opt/ros/humble/setup.bash
source /humble_ws/install/setup.bash

set -euxo pipefail

METRICS_ARGS=""

if [ ! -f /tmp/resim/inputs/batch_metrics_config.json ]; then
    MCAP_LOG_PATH=$(find /tmp/resim/inputs/logs -name "*.mcap" | head -n 1)

    if [ -z "$MCAP_LOG_PATH" ]; then
        echo "No .mcap log found in /tmp/resim/inputs/logs"
        exit 1
    fi

    if [ -f /tmp/resim/inputs/logs/ignore/emissions.ndjson ]; then
        mv /tmp/resim/inputs/logs/ignore/emissions.ndjson /tmp/resim/outputs/emissions.ndjson
    fi

    METRICS_ARGS="--log-path $MCAP_LOG_PATH"
fi

python3 metrics.py $METRICS_ARGS
