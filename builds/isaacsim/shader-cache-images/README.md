## Generating the shader cache for a new AMI or Isaac Sim version
Start up a batch in debug mode with the desired Isaac Sim image.
```
$ ./isaac-sim.sh --allow-root --/renderer/shadercache/driverDiskCache/enabled=true --/rtx/shaderDb/driverAppShaderCachePath=/shadercache --/rtx/shaderDb/driverAppShaderCacheDirPerDriver=true --/rtx/shaderDb/cachePermutationIndex=0
...
# wait for "Isaac Sim Full App is loaded." message

$ ./isaac-sim.sh --allow-root --/renderer/shadercache/driverDiskCache/enabled=true --/rtx/shaderDb/driverAppShaderCachePath=/shadercache --/rtx/shaderDb/driverAppShaderCacheDirPerDriver=true --/rtx/shaderDb/cachePermutationIndex=0
...
# run it again to check, should be ~20 seconds

$ cd /
$ tar czf shadercache.tar.gz shadercache/
$ tar tf shadercache.tar.gz 
shadercache/
shadercache/driver_575.57.08/
shadercache/driver_575.57.08/GLCache/
shadercache/driver_575.57.08/GLCache/7d216688b7134753963c32b58775ec90/
shadercache/driver_575.57.08/GLCache/7d216688b7134753963c32b58775ec90/ef1ebb9a439b1a99/
shadercache/driver_575.57.08/GLCache/7d216688b7134753963c32b58775ec90/ef1ebb9a439b1a99/eae91edf9a4c6d7a.bin
shadercache/driver_575.57.08/GLCache/7d216688b7134753963c32b58775ec90/ef1ebb9a439b1a99/eae91edf9a4c6d7a.toc
shadercache/driver_575.57.08/driverinfo.dat
$ mv shadercache.tar.gz /tmp/resim/outputs
```

Download the shader cache from the job's logs in the platform to a `build` directory under the Isaac version you're building for.
Copy it to s3 e.g.:
```
aws s3 cp build/shadercache.tar.gz s3://resim-binaries/isaac-cache/shadercache-4-2-0-resim-eks-gpu-20250814124908.tar.gz
```
Update the references in `rebuild_and_push_cache.sh` script to the new version.

Run `./rebuild_and_push_cache.sh` to rebuild the cache and push it to ECR.
