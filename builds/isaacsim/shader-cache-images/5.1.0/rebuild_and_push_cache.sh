#!/bin/bash

set -euxo pipefail

# change working directory parent of this script
pushd "$(dirname "$0")"

if [ ! -f "build/shadercache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/shadercache-5-1-0-resim-eks-gpu-20260217112256.tar.gz build/shadercache.tar.gz
fi

docker build . -f isaacsim.cache.dockerfile -t public.ecr.aws/resim/isaac-sim-resim-shaders:5.1.0-resim-eks-gpu-20260217112256 -t public.ecr.aws/resim/isaac-sim-resim-shaders:5.1.0
docker push public.ecr.aws/resim/isaac-sim-resim-shaders:5.1.0-resim-eks-gpu-20260217112256
docker push public.ecr.aws/resim/isaac-sim-resim-shaders:5.1.0

popd