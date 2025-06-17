#!/bin/bash

if [ ! -f "build/isaac-cache.tar.gz" ]; then
    mkdir -p build
    aws s3 cp s3://resim-binaries/isaac-cache/isaac-cache-575-4-5-0.tar.gz build/isaac-cache.tar.gz
fi

docker build . -f isaac.dockerfile -t isaac-humble-run --target run
docker build metrics -f metrics/Dockerfile -t isaac-humble-metrics
