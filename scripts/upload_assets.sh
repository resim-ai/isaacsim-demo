#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR/.."

aws --profile=rerun s3 sync assets/collected_hospital_demo s3://resim-binaries/demos/isaac_hospital/collected_hospital_demo/
aws --profile=rerun s3 sync assets/carter_warehouse_navigation_collected s3://resim-binaries/demos/isaac_warehouse/carter_warehouse_navigation_collected/

popd