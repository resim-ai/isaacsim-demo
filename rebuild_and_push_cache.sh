#!/bin/bash

set -euxo pipefail

if [ ! -f "build/isaac-cache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/isaac-cache-575-4-5-0.tar.gz build/isaac-cache.tar.gz
fi

docker build . -f builds/isaacsim/isaacsim.cache.dockerfile -t 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-shader-cache
docker push 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-shader-cache