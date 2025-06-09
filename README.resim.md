Build using 
```
./rebuild.sh
```

Run with [rocker](https://github.com/osrf/rocker)
```
sudo rocker --nvidia --x11 --privileged --network host --env="ACCEPT_EULA=Y" --env "OMNI_KIT_ALLOW_ROOT=1" --volume $PWD/outputs:/tmp/resim/outputs -- isaac-humble-run
```