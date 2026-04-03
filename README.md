# ReSim Isaac Sim Demos

This repository contains examples of using Isaac Sim with ReSim. It is based on the [IsaacSim-ros_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces) repo from NVIDIA Corporation, which is licensed under the Apache 2 license in [LICENSE.nvidia](LICENSE.nvidia). Any changes introduced by ReSim Inc. are licensed under the MIT license in the [LICENSE.resim](./LICENSE.resim) file.

## Moving to a new ReSim organization

Use this when you want the Isaac Sim demos (experiences, test suites, metrics, and cloud builds) in a **different ReSim org** or a fresh project.

### Prerequisites

1. **ReSim CLI** installed and authenticated for the **target** org (however your team switches orgs or API keys—do that before running any `resim` commands).
2. A **git checkout** of this repo on the commit whose images you intend to use (tags are derived from the current `HEAD` unless you override them—see below).
3. **Container images in ECR** for that commit: either rely on CI pushes to `909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images` (tags match `.github/workflows/docker-build.yml`), or build and push yourself with `./scripts/build_and_push.sh` after AWS/ECR login.

### How image tags and `docker-compose` fit together

CI publishes images to the **`isaacsim-test-images`** ECR repository with **short git SHA** suffixes, for example:

- `isaacsim-mcb-isaacsim-<short-sha>`
- `isaacsim-mcb-nav2-<short-sha>`
- `isaac-sim-metrics-<short-sha>`

[`builds/docker-compose.yml`](builds/docker-compose.yml) does not hard-code those URIs; it expects **`ISAACSIM_IMAGE`** and **`NAV2_IMAGE`** in the environment (same pattern as CI when registering a build).

From the repo root, load the standard URIs for the current commit:

```bash
source scripts/ecr_compose_env.sh
```

That exports `ISAACSIM_IMAGE`, `NAV2_IMAGE`, and `METRICS_IMAGE`, plus `COMMIT_SHA` (short) and `COMMIT_SHA_FULL` (full, for ReSim `--version` fields). To point at images for a **different** commit without checking it out, set `IMAGE_TAG_SHA` before sourcing (same short SHA as the ECR tag suffix). You can also override `ECR_REGISTRY_HOST` or `ECR_REPO` if your org uses another registry or repository name.

### Scripted bootstrap (starting point)

[`scripts/demo_in_new_org.sh`](scripts/demo_in_new_org.sh) is an end-to-end example: it creates a project and system, runs [`resim experiences sync`](./resim_experience_sync.yaml), registers metrics builds (including the Nav2 metrics image and the default reports image), registers an Isaac Sim **build** from [`builds/docker-compose.yml`](builds/docker-compose.yml) with `--use-os-env`, wires **Demo Smoke** to the Nav2 metrics build, and kicks off a test suite run.

Before running it:

- Edit **`PROJECT_NAME`** (and any suite names or asset IDs if your org’s setup differs).
- Ensure you are on the CLI’s **target org** and that the ECR tags for `COMMIT_SHA` **exist** (or push them first).

Run from the repository root:

```bash
./scripts/demo_in_new_org.sh
```

The script will fail early if `COMMIT_SHA_FULL` is missing (no git metadata); fix by using a proper clone or set `COMMIT_SHA_FULL` manually for ReSim version fields.

### Manual checklist (same shape as the script)

If you prefer not to use the script, mirror these steps with the ReSim CLI:

1. `resim projects create` — create the destination project.
2. `resim system create` — create the **Isaac Sim** system (adjust GPUs / memory / vCPUs as needed).
3. `resim experiences sync` — apply [`resim_experience_sync.yaml`](./resim_experience_sync.yaml) so experiences and managed suites exist.
4. `resim metrics-builds create` — register the Nav2 metrics image (`METRICS_IMAGE` after sourcing `ecr_compose_env.sh`) and optionally the default reports image (see the script for the public image URI).
5. `source scripts/ecr_compose_env.sh` then `resim builds create` with `--build-spec ./builds/docker-compose.yml` and **`--use-os-env`** so `ISAACSIM_IMAGE` / `NAV2_IMAGE` are picked up.
6. `resim suites revise` — attach the metrics build and metrics set to the suite you plan to run (e.g. **Demo Smoke**).
7. `resim test-suites run` — run a batch against that build (see [`.resim/metrics/config.resim.yml`](.resim/metrics/config.resim.yml) for metrics sync flags used in automation).

### Pushing images from your machine

If CI has not yet published tags for your commit, use AWS credentials that can push to the ECR repo and run:

```bash
./scripts/build_and_push.sh
```

Optional flags: `--nav2`, `--metrics`, `--isaacsim`, or `--all` (default). Override the tag suffix with `IMAGE_TAG_SHA=...` if you need a specific short SHA.

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

Build the images using:

```
./scripts/rebuild.sh
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
