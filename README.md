# ReSim Isaac Sim Demos

This repository contains examples of using Isaac Sim with ReSim. It is based on the [IsaacSim-ros_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces) repo from NVIDIA Corporation, which is licensed under the Apache 2 license in [LICENSE.nvidia](LICENSE.nvidia). Any changes introduced by ReSim Inc. are licensed under the MIT license in the [LICENSE.resim](./LICENSE.resim) file.

## Moving to a new ReSim organization

Use this when you want the Isaac Sim demos (experiences, test suites, metrics, and cloud builds) in a **different ReSim org** or a fresh project.

### Prerequisites

1. **ReSim CLI** available on your `PATH` and authenticated for the **target** org—or run [`scripts/demo_in_new_org.sh`](scripts/demo_in_new_org.sh), which **installs** the CLI (and **yq**) into `~/.local/bin` when missing, using **curl** or **wget**. Add that directory to your `PATH` in new shells, or set **`DEMO_LOCAL_BIN`** to another writable directory.
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

[`scripts/demo_in_new_org.sh`](scripts/demo_in_new_org.sh) is an end-to-end example: it **upserts** the project (create if missing; the CLI cannot update project metadata) and system (**`resim system update`** when **Isaac Sim** already exists), then registers **metrics builds** (Nav2 metrics image + default reports image) so **`test-suites create`** can attach **`--metrics-build`** / **`--metrics-set`** to **Demo Smoke**, **Warehouse Demo**, and **Hospital Demo** up front—no separate **`suites revise`**. It then reconciles experiences and **managed test suites** from [`resim_experience_sync.yaml`](./resim_experience_sync.yaml) in three steps: (1) **`resim experiences sync`** with `managedTestSuites` cleared so experiences exist, (2) **`resim test-suites create`** for missing `managedTestSuites` entries (those three suites get the Nav2 metrics build and **Nav2 Metrics** set), and (3) a full **`resim experiences sync`**. Finally it **`resim assets create`** for the two Isaac demo bundles if missing, **`resim builds create`** with `--use-os-env` and **`--assets`**, and two **`resim test-suites run`** calls: **Warehouse Demo** then **Hospital Demo** (override with **`WAREHOUSE_DEMO_SUITE_NAME`** / **`HOSPITAL_DEMO_SUITE_NAME`** if needed). It writes **`.demo_in_new_org_state`** (gitignored) with **`PROJECT_NAME`**, **`RESIM_URL`**, the ReSim **`projectID`** (so the same project *name* in a **different org** does not reuse another org’s IDs), and the registered metrics-build and Isaac **build** IDs for **`COMMIT_SHA_FULL`**. Reuse only happens when **all** of those match the current run (same project UUID, same staging/prod API, same git commit). State files **without** **`DEMO_PROJECT_ID`** (older format) are never reused. If you **change `PROJECT_NAME`**, switch **`--staging` / `--prod`**, sign into another **org**, or use a stale file, reuse is skipped and **new** registrations run; the file is **overwritten** on success. Delete **`DEMO_STATE_FILE`** / **`.demo_in_new_org_state`** anytime to force a clean pin. **`FORCE_NEW_BUILDS=1`** skips reuse even when everything matches.

Before running it:

- Edit **`PROJECT_NAME`** (and any suite names or asset IDs if your org’s setup differs).
- Choose **staging vs production** for the ReSim API (required). The script sets the same variables the CLI uses: **`RESIM_URL`** and **`RESIM_AUTH_URL`** (equivalent to global flags `--url` / `--auth-url`; see `resim --help`).
- Ensure you are signed in to the correct **org** for that environment and that the ECR tags for `COMMIT_SHA` **exist** (or push them first).

Run from the repository root:

```bash
./scripts/demo_in_new_org.sh --staging   # api.resim.io + resim-dev Auth0
./scripts/demo_in_new_org.sh --prod      # api.resim.ai + resim Auth0
```

The flag always sets `RESIM_URL` and `RESIM_AUTH_URL` for that run. Run `./scripts/demo_in_new_org.sh --help` for the exact URLs.

The script will fail early if `COMMIT_SHA_FULL` is missing (no git metadata); fix by using a proper clone or set `COMMIT_SHA_FULL` manually for ReSim version fields.

### Manual checklist (same shape as the script)

If you prefer not to use the script, mirror these steps with the ReSim CLI:

1. `resim projects create` — create the destination project.
2. `resim system create` — create the **Isaac Sim** system (adjust GPUs / memory / vCPUs as needed).
3. `resim metrics-builds create` — register the Nav2 metrics image (`METRICS_IMAGE` after sourcing `ecr_compose_env.sh`) and optionally the default reports image (see the script for the public image URI). Do this **before** creating test suites that reference a metrics build.
4. `resim experiences sync` — use the same three-phase managed-suite flow as [`scripts/demo_in_new_org.sh`](scripts/demo_in_new_org.sh); when you **`resim test-suites create`** **Demo Smoke**, **Warehouse Demo**, and **Hospital Demo**, pass **`--metrics-build`** and **`--metrics-set`** (`Nav2 Metrics`) so you do not need a later **`suites revise`**.
5. `resim assets create` — for each asset referenced by **`resim builds create --assets`** (here **`collected_hospital_demo`** and **`carter_warehouse_navigation_collected`**), with **locations** (S3) and **`--mount-folder`** set to the **leaf name only** (e.g. `collected_hospital_demo`); ReSim mounts under **`/tmp/resim/assets/<mount-folder>`**, so do not pass the full **`/tmp/resim/assets/...`** path as the mount folder.
6. `source scripts/ecr_compose_env.sh` then `resim builds create` with `--build-spec ./builds/docker-compose.yml`, **`--use-os-env`**, and **`--assets collected_hospital_demo,carter_warehouse_navigation_collected`** (latest revision each), or pin with **`name:<revision>`** where `<revision>` is the API asset revision—not UI “version 0”. **`name:0` is usually wrong** for newly created assets and can leave Isaac Sim without staged files, so **`LoadWorld`** fails for paths under **`/tmp/resim/assets/...`**.
7. `resim test-suites run` — run batches against that build for **Warehouse Demo** and **Hospital Demo** (see [`.resim/metrics/config.resim.yml`](.resim/metrics/config.resim.yml) for metrics sync flags used in automation).

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
