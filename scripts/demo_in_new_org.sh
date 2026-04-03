#!/bin/bash

set -euo pipefail

# ReSim CLI reads API and Auth0 endpoints from persistent flags --url / --auth-url, or the
# same values via env (prefix RESIM_, kebab→snake): RESIM_URL, RESIM_AUTH_URL.
# See: https://github.com/resim-ai/api-client/blob/main/cmd/resim/commands/client.go

usage() {
	cat <<'EOF' >&2
Usage: demo_in_new_org.sh --staging | --prod

  --staging  Staging: RESIM_URL=https://api.resim.io/v1/
             RESIM_AUTH_URL=https://resim-dev.us.auth0.com/

  --prod     Production: RESIM_URL=https://api.resim.ai/v1/
             RESIM_AUTH_URL=https://resim.us.auth0.com/

These match the ReSim CLI defaults for prod and the api-client e2e "staging" config.

Optional: GovCloud prod uses different URLs; run: resim govcloud enable (see resim --help).
For other endpoints, run the same resim commands yourself with --url / --auth-url (or export
RESIM_URL / RESIM_AUTH_URL) instead of this script.
EOF
}

RESIM_ENV=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		--staging)
			RESIM_ENV=staging
			shift
			;;
		--prod)
			RESIM_ENV=prod
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			echo "demo_in_new_org.sh: unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
done

if [[ -z "${RESIM_ENV}" ]]; then
	echo "demo_in_new_org.sh: specify --staging or --prod" >&2
	usage
	exit 1
fi

if [[ "${RESIM_ENV}" == staging ]]; then
	export RESIM_URL="https://api.resim.io/v1/"
	export RESIM_AUTH_URL="https://resim-dev.us.auth0.com/"
else
	export RESIM_URL="https://api.resim.ai/v1/"
	export RESIM_AUTH_URL="https://resim.us.auth0.com/"
fi

echo "Using ReSim environment: ${RESIM_ENV} (RESIM_URL=${RESIM_URL} RESIM_AUTH_URL=${RESIM_AUTH_URL})"

LOCAL_BIN="${DEMO_LOCAL_BIN:-${HOME}/.local/bin}"
mkdir -p "${LOCAL_BIN}"
case ":${PATH}:" in
*":${LOCAL_BIN}:"*) ;;
*) export PATH="${LOCAL_BIN}:${PATH}" ;;
esac

download_to() {
	local url="$1"
	local dest="$2"
	if command -v curl >/dev/null 2>&1; then
		curl -fL --retry 3 --connect-timeout 15 "$url" -o "$dest"
	elif command -v wget >/dev/null 2>&1; then
		wget -q --timeout=15 -O "$dest" "$url"
	else
		echo "demo_in_new_org.sh: need curl or wget to download tools" >&2
		exit 1
	fi
}

detect_resim_platform() {
	local os arch
	os=$(uname -s | tr '[:upper:]' '[:lower:]')
	arch=$(uname -m)
	case "${arch}" in
	x86_64) arch=amd64 ;;
	aarch64 | arm64) arch=arm64 ;;
	*)
		echo "demo_in_new_org.sh: unsupported machine $(uname -m); install resim manually" >&2
		return 1
		;;
	esac
	case "${os}" in
	linux | darwin) printf '%s %s' "${os}" "${arch}" ;;
	*)
		echo "demo_in_new_org.sh: unsupported OS ${os}; install resim manually" >&2
		return 1
		;;
	esac
}

ensure_resim_cli() {
	if command -v resim >/dev/null 2>&1; then
		return 0
	fi
	echo "Installing ReSim CLI into ${LOCAL_BIN} ..."
	local os arch tmp plat
	plat="$(detect_resim_platform)" || exit 1
	read -r os arch <<<"${plat}"
	tmp="$(mktemp)"
	download_to "https://github.com/resim-ai/api-client/releases/latest/download/resim-${os}-${arch}" "${tmp}"
	chmod +x "${tmp}"
	mv "${tmp}" "${LOCAL_BIN}/resim"
	echo "Installed: $(command -v resim)"
}

ensure_yq() {
	if command -v yq >/dev/null 2>&1; then
		return 0
	fi
	echo "Installing yq (mikefarah) into ${LOCAL_BIN} ..."
	local os arch tmp plat
	plat="$(detect_resim_platform)" || exit 1
	read -r os arch <<<"${plat}"
	tmp="$(mktemp)"
	download_to "https://github.com/mikefarah/yq/releases/latest/download/yq_${os}_${arch}" "${tmp}"
	chmod +x "${tmp}"
	mv "${tmp}" "${LOCAL_BIN}/yq"
	echo "Installed: $(command -v yq)"
}

ensure_resim_cli
ensure_yq

PROJECT_NAME="Iain's Isaac Sim Sandbox"
PROJECT_DESCRIPTION="A project for running the Nav2 Demo"
SYSTEM_NAME="Isaac Sim"
SYSTEM_DESCRIPTION="A system for running Isaac Sim"
# Test suite that runs the Nav2 demo batch; created with Nav2 metrics build + set (no later suites revise).
DEMO_SMOKE_SUITE_NAME="${DEMO_SMOKE_SUITE_NAME:-Demo Smoke}"
NAV2_METRICS_SET_NAME="${NAV2_METRICS_SET_NAME:-Nav2 Metrics}"
EXPERIENCES_CONFIG="${EXPERIENCES_CONFIG:-./resim_experience_sync.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ecr_compose_env.sh
source "${SCRIPT_DIR}/ecr_compose_env.sh"

# ReSim --version fields use full git SHA (same as CI).
if [ -z "${COMMIT_SHA_FULL}" ]; then
	echo "demo_in_new_org.sh: need a git checkout (or set COMMIT_SHA_FULL) for ReSim build/metrics versions" >&2
	exit 1
fi
METRICS_VERSION="${METRICS_VERSION:-${COMMIT_SHA_FULL}}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
if [[ "${EXPERIENCES_CONFIG}" != /* ]]; then
	EXPERIENCES_CONFIG="${REPO_ROOT}/${EXPERIENCES_CONFIG#./}"
fi

ensure_project() {
	if resim projects get --project "${PROJECT_NAME}" &>/dev/null; then
		echo "Project already exists: ${PROJECT_NAME}"
		return 0
	fi
	resim projects create --name "${PROJECT_NAME}" --description "${PROJECT_DESCRIPTION}"
}

ensure_system() {
	local -a sys_resources=(
		--build-gpus 1
		--build-memory-mib 32768
		--build-vcpus 8
		--metrics-build-vcpus 2
		--metrics-build-memory-mib 8192
	)
	if resim system get --project "${PROJECT_NAME}" --system "${SYSTEM_NAME}" &>/dev/null; then
		echo "System already exists: updating ${SYSTEM_NAME}..."
		resim system update --project "${PROJECT_NAME}" --system "${SYSTEM_NAME}" \
			--name "${SYSTEM_NAME}" --description "${SYSTEM_DESCRIPTION}" \
			"${sys_resources[@]}"
		return 0
	fi
	resim system create --project "${PROJECT_NAME}" \
		--name "${SYSTEM_NAME}" --description "${SYSTEM_DESCRIPTION}" \
		"${sys_resources[@]}"
}

# `resim experiences sync` revises managed test suites but requires each suite to already exist
# (see api-client sync/test_suites.go). Sync once with managedTestSuites cleared so experiences
# exist, create any missing suites from the YAML, then sync again with the full config.
sync_experiences_without_managed_suites() (
	tmp="$(mktemp)"
	trap 'rm -f "${tmp}"' EXIT
	yq '.managedTestSuites = []' "${EXPERIENCES_CONFIG}" >"${tmp}"
	echo "Syncing experiences (without managed test suites) from ${EXPERIENCES_CONFIG}..."
	resim experiences sync --project "${PROJECT_NAME}" --experiences-config "${tmp}"
)

ensure_managed_test_suites_from_config() {
	local n i name desc exp_csv
	n="$(yq '(.managedTestSuites // []) | length' "${EXPERIENCES_CONFIG}")"
	if [[ "${n}" == "0" || -z "${n}" ]]; then
		echo "No managedTestSuites in config; skipping test suite creation."
		return 0
	fi
	echo "Ensuring ${n} managed test suite(s) exist (from managedTestSuites in ${EXPERIENCES_CONFIG})..."
	for ((i = 0; i < n; i++)); do
		name="$(yq -r ".managedTestSuites[${i}].name" "${EXPERIENCES_CONFIG}")"
		if [[ -z "${name}" || "${name}" == "null" ]]; then
			echo "demo_in_new_org.sh: managedTestSuites[${i}] missing name" >&2
			exit 1
		fi
		if resim test-suites get --project "${PROJECT_NAME}" --test-suite "${name}" &>/dev/null; then
			echo "  Test suite exists: ${name}"
			continue
		fi
		exp_csv="$(yq -r ".managedTestSuites[${i}].experiences | join(\",\")" "${EXPERIENCES_CONFIG}")"
		if [[ -z "${exp_csv}" || "${exp_csv}" == "null" ]]; then
			echo "demo_in_new_org.sh: managed test suite \"${name}\" has no experiences" >&2
			exit 1
		fi
		printf -v desc 'Managed test suite "%s" (from experience sync config)' "${name}"
		echo "  Creating test suite: ${name}"
		local -a create_cmd=(
			resim test-suites create --project "${PROJECT_NAME}" --system "${SYSTEM_NAME}"
			--name "${name}" --description "${desc}" --experiences "${exp_csv}"
		)
		if [[ "${name}" == "${DEMO_SMOKE_SUITE_NAME}" ]]; then
			if [[ -z "${NAV2_METRICS_BUILD_ID:-}" ]]; then
				echo "demo_in_new_org.sh: NAV2_METRICS_BUILD_ID must be set before creating ${DEMO_SMOKE_SUITE_NAME}" >&2
				exit 1
			fi
			create_cmd+=(--metrics-build "${NAV2_METRICS_BUILD_ID}" --metrics-set "${NAV2_METRICS_SET_NAME}")
		fi
		"${create_cmd[@]}"
	done
}

sync_experiences_full_config() {
	echo "Syncing experiences and managed test suites from ${EXPERIENCES_CONFIG}..."
	resim experiences sync --project "${PROJECT_NAME}" --experiences-config "${EXPERIENCES_CONFIG}"
}

# Shared S3 Isaac assets (same URIs as ReSim’s reference org). builds create --assets expects these names.
# Use the latest asset revision (omit ":<n>" on --assets). The CLI treats "name:0" as revision *zero*,
# not "version 1"; pinning :0 often yields no usable files on disk inside Isaac Sim, so LoadWorld fails
# with "Could not find path '/tmp/resim/assets/...'" (paths in builds/nav2/experiences must exist on the
# isaacsim container after ReSim stages linked assets).
# mount-folder is the leaf directory name only; ReSim prepends /tmp/resim/assets/ (do not pass the full path).
ensure_asset() {
	local name="$1" desc="$2" locations="$3" mount="$4" version="${5:-1}"
	if resim assets get --project "${PROJECT_NAME}" --asset "${name}" &>/dev/null; then
		echo "Asset already exists: ${name}"
		return 0
	fi
	echo "Creating asset: ${name}"
	resim assets create --project "${PROJECT_NAME}" --name "${name}" --description "${desc}" \
		--locations "${locations}" --mount-folder "${mount}" --version "${version}"
}

ensure_demo_assets() {
	ensure_asset "collected_hospital_demo" \
		"Hospital demo scenes and props." \
		"s3://resim-binaries/demos/isaac_hospital/collected_hospital_demo/" \
		"collected_hospital_demo"
	ensure_asset "carter_warehouse_navigation_collected" \
		"Scenes & props for Warehouse scenarios." \
		"s3://resim-binaries/demos/isaac_warehouse/carter_warehouse_navigation_collected/" \
		"carter_warehouse_navigation_collected"
}

# Function to pause execution
pause() {
    echo "Press Enter to continue to the next command..."
    read -p ""
}

echo "Signed in to the right ReSim org for ${RESIM_ENV}? (CLI uses RESIM_URL / RESIM_AUTH_URL above.)"
read -p "Press Enter to continue"

# Upsert project and system (CLI has no projects update; existing project is left as-is).
ensure_project
# pause

ensure_system
# pause

# Metrics builds before managed test suites so Demo Smoke can be created with --metrics-build / --metrics-set.
_nav2_mb_out="$(
	resim metrics-builds create --project "${PROJECT_NAME}" \
		--name "Nav2 Metrics" \
		--image "${METRICS_IMAGE}" \
		--version "${METRICS_VERSION}" \
		--systems "${SYSTEM_NAME}" \
		--github
)"
echo "${_nav2_mb_out}"
export NAV2_METRICS_BUILD_ID="$(printf '%s' "${_nav2_mb_out}" | grep -oE 'metrics_build_id=[a-f0-9-]{36}' | head -1 | cut -d= -f2)"
if [[ -z "${NAV2_METRICS_BUILD_ID}" ]]; then
	export NAV2_METRICS_BUILD_ID="$(printf '%s' "${_nav2_mb_out}" | sed -n 's/.*metrics_build_id=\([a-f0-9-]*\).*/\1/p' | head -1)"
fi
if [[ -z "${NAV2_METRICS_BUILD_ID}" ]]; then
	echo "demo_in_new_org.sh: metrics-builds create (Nav2) did not print metrics_build_id=..." >&2
	exit 1
fi
# pause

_default_rep_out="$(
	resim metrics-builds create --project "${PROJECT_NAME}" \
		--name "Default Report Metrics Build" \
		--image public.ecr.aws/resim/open-metrics-builds/default-reports-build:sha-b64ef17bcd45d62bf0c1a09cad168372ffe87e9c \
		--version "b64ef17bcd45d62bf0c1a09cad168372ffe87e9c" \
		--github
)"
echo "${_default_rep_out}"
export DEFAULT_REPORT_METRICS_BUILD_ID="$(printf '%s' "${_default_rep_out}" | grep -oE 'metrics_build_id=[a-f0-9-]{36}' | head -1 | cut -d= -f2)"
if [[ -z "${DEFAULT_REPORT_METRICS_BUILD_ID}" ]]; then
	export DEFAULT_REPORT_METRICS_BUILD_ID="$(printf '%s' "${_default_rep_out}" | sed -n 's/.*metrics_build_id=\([a-f0-9-]*\).*/\1/p' | head -1)"
fi
if [[ -z "${DEFAULT_REPORT_METRICS_BUILD_ID}" ]]; then
	echo "demo_in_new_org.sh: metrics-builds create (default reports) did not print metrics_build_id=..." >&2
	exit 1
fi
# pause

# Experiences first (managedTestSuites empty), then create missing suites (Demo Smoke includes metrics), then full sync.
sync_experiences_without_managed_suites
ensure_managed_test_suites_from_config
sync_experiences_full_config
# pause

# builds create --assets resolves names to project assets; create them first in a fresh org.
ensure_demo_assets
# pause

# Set up the system build (ISAACSIM_IMAGE / NAV2_IMAGE / METRICS_IMAGE from ecr_compose_env.sh)
_build_out="$(
	resim builds create --project "${PROJECT_NAME}" --build-spec ./builds/docker-compose.yml \
		--system "${SYSTEM_NAME}" --name "Isaac Sim Build @ ${COMMIT_SHA}" --description "Isaac Sim Nav2 demo build" \
		--branch "main" --version "${COMMIT_SHA_FULL}" --auto-create-branch --github --use-os-env \
		--assets "collected_hospital_demo,carter_warehouse_navigation_collected"
)"
echo "${_build_out}"
export ISAAC_SIM_BUILD_ID="$(printf '%s' "${_build_out}" | grep -oE 'build_id=[a-f0-9-]{36}' | head -1 | cut -d= -f2)"
if [[ -z "${ISAAC_SIM_BUILD_ID}" ]]; then
	echo "demo_in_new_org.sh: builds create did not print build_id=... (see output above)" >&2
	exit 1
fi
# pause

# Run the batch
resim test-suites run --project "${PROJECT_NAME}" \
  --test-suite "${DEMO_SMOKE_SUITE_NAME}" \
  --batch-name "Nav2 Demo" \
  --build-id "${ISAAC_SIM_BUILD_ID}" \
  --pool-labels 'resim:metrics2:k8s' \
  --sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml \
  --allowable-failure-percent 25
# pause

echo "Once the above batch is complete, run the report using the following command (same API/auth as this run):"
printf 'export RESIM_URL=%q RESIM_AUTH_URL=%q\n' "${RESIM_URL}" "${RESIM_AUTH_URL}"
echo "resim reports create --branch main --metrics-build-id $DEFAULT_REPORT_METRICS_BUILD_ID --project \"${PROJECT_NAME}\" --test-suite \"${DEMO_SMOKE_SUITE_NAME}\" --length 1"
