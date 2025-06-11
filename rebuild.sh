docker build . -f isaac.dockerfile -t isaac-humble-run --target run
docker build metrics -f metrics/Dockerfile -t isaac-humble-metrics
