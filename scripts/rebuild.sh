#!/bin/bash

set -euxo pipefail

docker build . -f builds/isaacsim/isaacsim.dockerfile -t resim-isaacsim
docker build . -f builds/nav2/nav2.dockerfile -t resim-humble-nav2
docker build . -f metrics/Dockerfile -t isaac-humble-metrics
