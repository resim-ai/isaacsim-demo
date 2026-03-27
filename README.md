# ReSim Isaac Sim Demos

This repository contains examples of using Isaac Sim with ReSim. It is based on the [IsaacSim-ros_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces) repo from NVIDIA Corporation, which is licensed under the Apache 2 license in [LICENSE.nvidia](LICENSE.nvidia). Any changes introduced by ReSim Inc. are licensed under the MIT license in the [LICENSE.resim](./LICENSE.resim) file.

## Shader Cache

This repo contains examples of building a shader cache to optimise startup times when running Isaac Sim in the cloud. See `builds/isaacsim/shader-cache-images` for more details.

## Examples

### ROS Nav2

This example is based on the [ROS2 Navigation](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_navigation.html) example from the NVIDIA docs.

#### Setup

Download the assets:
```
./scripts/download_assets.sh
```

Install ROS2 Humble.

#### Running

Start up Isaac Sim 5.1.0 with these arguments:
```
./isaac-sim.sh --enable isaacsim.replicator.agent.core --enable omni.anim.graph.core --enable isaacsim.ros2.sim_control --/app/scripting/ignoreWarningDialog=true
```

Then start the stack, providing a path to an experience from `builds/nav2/experiences`:
```
./scripts/run_demo.sh ${NAME_OF_EXPERIENCE}
```

#### Building docker images

Build the images using 
```
./rebuild.sh
```

Run on a system with a GPU:
```
docker compose -f builds/docker-compose.local.yml up --abort-on-container-exit
```

Run the metrics:
```
docker run -v ./builds/outputs:/tmp/resim/inputs/logs -v ./outputs:/tmp/resim/outputs isaac-humble-metrics
```

The resulting `metrics.binproto` can be visualised in the ReSim metrics debugger at `https://app.resim.ai/metrics-debugger`
