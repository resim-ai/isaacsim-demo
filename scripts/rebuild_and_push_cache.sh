#!/bin/bash

set -euxo pipefail

if [ ! -f "build/shadercache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/shadercache-575-5-0-0.tar.gz build/shadercache.tar.gz
fi

docker build . -f builds/isaacsim/isaacsim.cache.dockerfile -t 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-5-0-0-shader-cache
docker push 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaacsim-mcb-isaacsim-5-0-0-shader-cache