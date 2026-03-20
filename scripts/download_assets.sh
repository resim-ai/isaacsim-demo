#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR/.."

aws --profile=rerun s3 sync s3://resim-binaries/demos/isaac_hospital/collected_hospital_demo/ assets/collected_hospital_demo 

popd