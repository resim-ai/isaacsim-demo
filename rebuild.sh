#!/bin/bash

set -euxo pipefail

if [ ! -f "build/isaac-cache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/isaac-cache-575-4-5-0.tar.gz build/isaac-cache.tar.gz
fi

docker build . -f builds/isaacsim/isaacsim.dockerfile -t resim-isaacsim
docker build . -f builds/nav2/nav2.dockerfile -t resim-humble-nav2
docker build metrics -f metrics/Dockerfile -t isaac-humble-metrics

docker tag resim-isaacsim 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:isaacsim-mcb-isaacsim
docker tag resim-humble-nav2 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:isaacsim-mcb-nav2
docker tag isaac-humble-metrics 909785973729.dkr.ecr.us-east-1.amazonaws.com/customer-test-images:isaacsim-mcb-metrics