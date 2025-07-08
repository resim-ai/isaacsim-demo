## Shader Cache

The shader cache is a pre-built cache of the baseline shaders needed for Isaac Sim.
These are roughly unique to the Isaac Sim version and NVIDIA driver version.
To generate a new cache, in the event of an update of either of these, start a [debug mode](https://docs.resim.ai/guides/debug/) job, and run the following commands:

```
rm -rf /isaac-sim/kit/cache/ /root/.cache/ov /root/.nv/ComputeCache
./warmup.sh
tar czf /tmp/resim/outputs/isaac-cache.tar.gz /isaac-sim/kit/cache/ /root/.cache/ov /root/.nv/ComputeCache
```
Exit the debug job, and download the cache from the job's logs in the platform.
Upload the cache to S3:

```
aws s3 cp isaac-cache.tar.gz s3://resim-binaries/isaac-cache/isaac-cache-{driver-version}-{isaac-sim-version}.tar.gz
```

and update the reference to the cache in `isaacsim.cache.dockerfile`.
Run `./rebuild_and_push_cache.sh` to rebuild the cache and push it to ECR.

## Builds

Build using 
```
./rebuild.sh
```

Run on a system with a GPU:
```
docker compose -f builds/docker-compose.yml up --abort-on-container-exit
```