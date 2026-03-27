#!/bin/bash

set -euxo pipefail

# change working directory parent of this script
pushd "$(dirname "$0")"

if [ ! -f "build/shadercache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/shadercache-4-2-0-resim-eks-gpu-20260320191933.tar.gz build/shadercache.tar.gz
fi

docker build . -f isaacsim.cache.dockerfile -t public.ecr.aws/resim/isaac-sim-resim-shaders:4.2.0-resim-eks-gpu-20260320191933 -t public.ecr.aws/resim/isaac-sim-resim-shaders:4.2.0
docker push public.ecr.aws/resim/isaac-sim-resim-shaders:4.2.0-resim-eks-gpu-20260320191933
docker push public.ecr.aws/resim/isaac-sim-resim-shaders:4.2.0

popd