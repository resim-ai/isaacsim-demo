#!/bin/bash

set -euo pipefail

# ReSim CLI reads API and Auth0 endpoints from persistent flags --url / --auth-url, or the
# same values via env (prefix RESIM_, kebab→snake): RESIM_URL, RESIM_AUTH_URL.
# See: https://github.com/resim-ai/api-client/blob/main/cmd/resim/commands/client.go

usage() {
	cat <<'EOF' >&2
Usage: demo_in_new_org.sh --staging | --prod [--yes]

  --staging  Staging: RESIM_URL=https://api.resim.io/v1/
             RESIM_AUTH_URL=https://resim-dev.us.auth0.com/

  --prod     Production: RESIM_URL=https://api.resim.ai/v1/
             RESIM_AUTH_URL=https://resim.us.auth0.com/

These match the ReSim CLI defaults for prod and the api-client e2e "staging" config.

Optional: GovCloud prod uses different URLs; run: resim govcloud enable (see resim --help).
For other endpoints, run the same resim commands yourself with --url / --auth-url (or export
RESIM_URL / RESIM_AUTH_URL) instead of this script.

Build registration: metrics-builds + Isaac build are recreated when git HEAD, PROJECT_NAME,
RESIM_URL, or the ReSim project UUID no longer matches DEMO_STATE_FILE. The state file stores
DEMO_PROJECT_ID so the same project *name* in another org does not reuse another org's build IDs.
If the state file is stale (wrong project name, API URL, project ID, commit, or missing
DEMO_PROJECT_ID), it is deleted and in-memory pins cleared so no old build IDs leak from disk
or from a parent shell. FORCE_NEW_BUILDS=1 always re-registers.

  --yes, -y  Skip the "Press Enter to continue" org check (also skipped when stdin is not a TTY).
             Or set DEMO_ASSUME_YES=1 before running.
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
		--yes | -y)
			DEMO_ASSUME_YES=1
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

PROJECT_NAME="Isaac Sim Sandbox"
PROJECT_DESCRIPTION="A project for running the Nav2 Demo"
SYSTEM_NAME="Isaac Sim"
SYSTEM_DESCRIPTION="A system for running Isaac Sim"
# Managed suites: smoke (optional manual use), warehouse + hospital batches kicked off at end of script.
DEMO_SMOKE_SUITE_NAME="${DEMO_SMOKE_SUITE_NAME:-Demo Smoke}"
WAREHOUSE_DEMO_SUITE_NAME="${WAREHOUSE_DEMO_SUITE_NAME:-Warehouse Demo}"
HOSPITAL_DEMO_SUITE_NAME="${HOSPITAL_DEMO_SUITE_NAME:-Hospital Demo}"
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

STATE_FILE="${DEMO_STATE_FILE:-${REPO_ROOT}/.demo_in_new_org_state}"

# Avoid reusing IDs from an unrelated shell session (exported before this script ran).
unset NAV2_METRICS_BUILD_ID DEFAULT_REPORT_METRICS_BUILD_ID ISAAC_SIM_BUILD_ID 2>/dev/null || true

clear_demo_state_vars() {
	DEMO_PROJECT_NAME=""
	DEMO_RESIM_URL=""
	DEMO_PROJECT_ID=""
	DEMO_LAST_COMMIT_FULL=""
	DEMO_NAV2_METRICS_BUILD_ID=""
	DEMO_DEFAULT_REPORT_METRICS_BUILD_ID=""
	DEMO_ISAAC_SIM_BUILD_ID=""
}

# Drop stale on-disk state and clear sourced DEMO_* so we never mix orgs/projects/commits.
reset_demo_state_file() {
	local reason="$1"
	echo "Resetting demo state (${reason}): removing ${STATE_FILE}" >&2
	rm -f "${STATE_FILE}"
	clear_demo_state_vars
}

ensure_project() {
	echo "Ensuring project \"${PROJECT_NAME}\" (projects get)..."
	# Do not redirect stdout: ReSim prints device-login / auth instructions on stdout.
	if resim projects get --project "${PROJECT_NAME}"; then
		echo "Project already exists: ${PROJECT_NAME}"
		return 0
	fi
	echo "Creating project \"${PROJECT_NAME}\"..."
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
	echo "Ensuring system \"${SYSTEM_NAME}\" (system get)..."
	if resim system get --project "${PROJECT_NAME}" --system "${SYSTEM_NAME}"; then
		echo "System already exists: updating ${SYSTEM_NAME}..."
		resim system update --project "${PROJECT_NAME}" --system "${SYSTEM_NAME}" \
			--name "${SYSTEM_NAME}" --description "${SYSTEM_DESCRIPTION}" \
			"${sys_resources[@]}"
		return 0
	fi
	echo "Creating system \"${SYSTEM_NAME}\"..."
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

# Nav2 metrics build + set attached at suite create (no later suites revise).
managed_suite_uses_nav2_metrics() {
	local n="$1"
	[[ "${n}" == "${DEMO_SMOKE_SUITE_NAME}" ]] ||
		[[ "${n}" == "${WAREHOUSE_DEMO_SUITE_NAME}" ]] ||
		[[ "${n}" == "${HOSPITAL_DEMO_SUITE_NAME}" ]]
}

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
		if resim test-suites get --project "${PROJECT_NAME}" --test-suite "${name}"; then
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
		if managed_suite_uses_nav2_metrics "${name}"; then
			if [[ -z "${NAV2_METRICS_BUILD_ID:-}" ]]; then
				echo "demo_in_new_org.sh: NAV2_METRICS_BUILD_ID must be set before creating ${name}" >&2
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
	echo "Ensuring asset \"${name}\" (assets get)..."
	if resim assets get --project "${PROJECT_NAME}" --asset "${name}"; then
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

# resim projects get --project <name> emits JSON with .projectID (unique per ReSim project; differs across orgs for the same name).
get_current_project_id() {
	local json pid
	json="$(resim projects get --project "${PROJECT_NAME}")" || return 1
	pid="$(printf '%s' "${json}" | yq -r '.projectID // empty')"
	if [[ -z "${pid}" || "${pid}" == "null" ]]; then
		echo "demo_in_new_org.sh: could not parse .projectID from resim projects get output" >&2
		return 1
	fi
	printf '%s' "${pid}"
}

load_demo_state() {
	clear_demo_state_vars
	[[ -f "${STATE_FILE}" ]] || return 0
	# shellcheck disable=SC1090
	source "${STATE_FILE}"
}

# After ensure_project: if context does not match the state file, delete it (and clear vars).
# Handles new org, recreated project, staging↔prod, legacy files without DEMO_PROJECT_ID, and
# stray exports from the parent shell (cleared above + here when file is removed).
invalidate_stale_demo_state() {
	[[ -f "${STATE_FILE}" ]] || return 0
	if [[ -n "${DEMO_PROJECT_NAME:-}" && "${DEMO_PROJECT_NAME}" != "${PROJECT_NAME}" ]]; then
		reset_demo_state_file "project name \"${DEMO_PROJECT_NAME}\" != \"${PROJECT_NAME}\""
		return 0
	fi
	if [[ -n "${DEMO_RESIM_URL:-}" && "${DEMO_RESIM_URL}" != "${RESIM_URL}" ]]; then
		reset_demo_state_file "stored RESIM_URL != current API host"
		return 0
	fi
	if [[ -z "${DEMO_PROJECT_ID:-}" ]]; then
		reset_demo_state_file "missing DEMO_PROJECT_ID (legacy or copied state)"
		return 0
	fi
	if [[ -n "${DEMO_LAST_COMMIT_FULL:-}" && "${DEMO_LAST_COMMIT_FULL}" != "${COMMIT_SHA_FULL}" ]]; then
		reset_demo_state_file "git commit changed (${DEMO_LAST_COMMIT_FULL} -> ${COMMIT_SHA_FULL})"
		return 0
	fi
	local current_pid
	if ! current_pid="$(get_current_project_id)"; then
		echo "demo_in_new_org.sh: could not verify current project ID; leaving ${STATE_FILE} unchanged" >&2
		return 0
	fi
	if [[ "${current_pid}" != "${DEMO_PROJECT_ID}" ]]; then
		reset_demo_state_file "ReSim projectID mismatch (new org, new project, or recreated project)"
		return 0
	fi
}

try_reuse_registered_builds() {
	if [[ -n "${FORCE_NEW_BUILDS:-}" ]]; then
		return 1
	fi
	# IDs are scoped to a ReSim project and API host; never reuse across projects or orgs.
	[[ -n "${DEMO_PROJECT_NAME}" ]] || return 1
	[[ "${DEMO_PROJECT_NAME}" == "${PROJECT_NAME}" ]] || return 1
	[[ -n "${DEMO_RESIM_URL}" ]] || return 1
	[[ "${DEMO_RESIM_URL}" == "${RESIM_URL}" ]] || return 1
	if [[ -z "${DEMO_PROJECT_ID:-}" ]]; then
		echo "Not reusing ${STATE_FILE}: missing DEMO_PROJECT_ID (older state). Will register new metrics/builds for this org."
		return 1
	fi
	local current_pid
	current_pid="$(get_current_project_id)" || return 1
	if [[ "${current_pid}" != "${DEMO_PROJECT_ID}" ]]; then
		echo "Not reusing ${STATE_FILE}: state project ID ${DEMO_PROJECT_ID} != current project ${current_pid}."
		echo "  (Same project name in another ReSim org is a different project — delete ${STATE_FILE} or use FORCE_NEW_BUILDS=1.)"
		return 1
	fi
	[[ -n "${DEMO_LAST_COMMIT_FULL}" ]] || return 1
	[[ "${DEMO_LAST_COMMIT_FULL}" == "${COMMIT_SHA_FULL}" ]] || return 1
	[[ -n "${DEMO_NAV2_METRICS_BUILD_ID}" ]] || return 1
	[[ -n "${DEMO_DEFAULT_REPORT_METRICS_BUILD_ID}" ]] || return 1
	[[ -n "${DEMO_ISAAC_SIM_BUILD_ID}" ]] || return 1
	export NAV2_METRICS_BUILD_ID="${DEMO_NAV2_METRICS_BUILD_ID}"
	export DEFAULT_REPORT_METRICS_BUILD_ID="${DEMO_DEFAULT_REPORT_METRICS_BUILD_ID}"
	export ISAAC_SIM_BUILD_ID="${DEMO_ISAAC_SIM_BUILD_ID}"
	echo "Reusing ReSim metrics + Isaac build IDs for project \"${PROJECT_NAME}\" (projectID=${DEMO_PROJECT_ID}), commit ${COMMIT_SHA} (${COMMIT_SHA_FULL})."
	echo "  (Set FORCE_NEW_BUILDS=1 to call metrics-builds create + builds create anyway.)"
	return 0
}

save_demo_state() {
	local pin_project_id
	pin_project_id="$(get_current_project_id)" || exit 1
	umask 077
	{
		printf 'DEMO_PROJECT_NAME=%q\n' "${PROJECT_NAME}"
		printf 'DEMO_RESIM_URL=%q\n' "${RESIM_URL}"
		printf 'DEMO_PROJECT_ID=%q\n' "${pin_project_id}"
		printf 'DEMO_LAST_COMMIT_FULL=%q\n' "${COMMIT_SHA_FULL}"
		printf 'DEMO_NAV2_METRICS_BUILD_ID=%q\n' "${NAV2_METRICS_BUILD_ID}"
		printf 'DEMO_DEFAULT_REPORT_METRICS_BUILD_ID=%q\n' "${DEFAULT_REPORT_METRICS_BUILD_ID}"
		printf 'DEMO_ISAAC_SIM_BUILD_ID=%q\n' "${ISAAC_SIM_BUILD_ID}"
	} >"${STATE_FILE}.new"
	mv "${STATE_FILE}.new" "${STATE_FILE}"
	echo "Wrote ${STATE_FILE} (pin projectID=${pin_project_id}, API ${RESIM_URL}, commit ${COMMIT_SHA})."
}

# Optional step-through (uncomment # pause lines). Skipped with --yes / DEMO_ASSUME_YES=1 or non-TTY stdin.
pause() {
	if [[ -n "${DEMO_ASSUME_YES:-}" ]] || ! [[ -t 0 ]]; then
		return 0
	fi
	echo "Press Enter to continue to the next command..."
	read -r
}

echo "Signed in to the right ReSim org for ${RESIM_ENV}? (CLI uses RESIM_URL / RESIM_AUTH_URL above.)"
if [[ -n "${DEMO_ASSUME_YES:-}" ]] || ! [[ -t 0 ]]; then
	echo "Continuing without confirmation (--yes, DEMO_ASSUME_YES=1, or non-interactive stdin)."
else
	read -r -p "Press Enter to continue " || true
fi
echo ""
echo "Starting setup. The next steps call the ReSim API (${RESIM_URL}); the first request can take a while (auth, TLS, cold start)."

# Upsert project and system (CLI has no projects update; existing project is left as-is).
ensure_project
# pause

ensure_system
# pause

load_demo_state
invalidate_stale_demo_state
REGISTERED_BUILDS_REUSED=0
if try_reuse_registered_builds; then
	REGISTERED_BUILDS_REUSED=1
fi

# New git commit => new metrics-builds + new Isaac build. Same commit => reuse IDs from STATE_FILE.
if [[ "${REGISTERED_BUILDS_REUSED}" -eq 0 ]]; then
	# Metrics builds before managed test suites so suites that use Nav2 metrics can be created with --metrics-build / --metrics-set.
	echo "Registering Nav2 metrics-build (metrics-builds create; image ${METRICS_IMAGE})..."
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

	echo "Registering default report metrics-build..."
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
fi

# Experiences first (managedTestSuites empty), then create missing suites (Demo Smoke / Warehouse / Hospital get Nav2 metrics), then full sync.
sync_experiences_without_managed_suites
ensure_managed_test_suites_from_config
sync_experiences_full_config
# pause

# builds create --assets resolves names to project assets; create them first in a fresh org.
ensure_demo_assets
# pause

if [[ "${REGISTERED_BUILDS_REUSED}" -eq 0 ]]; then
	# Set up the system build (ISAACSIM_IMAGE / NAV2_IMAGE / METRICS_IMAGE from ecr_compose_env.sh)
	echo "Creating Isaac Sim system build (builds create; ECR images for commit ${COMMIT_SHA})..."
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
	save_demo_state
	# pause
fi

# Catch stale pins (e.g. build deleted in UI) before test-suites run / metrics sync.
ensure_isaac_system_build_exists() {
	[[ -n "${ISAAC_SIM_BUILD_ID:-}" ]] || {
		echo "demo_in_new_org.sh: ISAAC_SIM_BUILD_ID is empty" >&2
		exit 1
	}
	echo "Verifying system build exists (builds get --build-id ${ISAAC_SIM_BUILD_ID})..."
	if ! resim builds get --project "${PROJECT_NAME}" --build-id "${ISAAC_SIM_BUILD_ID}" >/dev/null; then
		echo "demo_in_new_org.sh: system build not found for this project. Try FORCE_NEW_BUILDS=1 or delete ${STATE_FILE}." >&2
		exit 1
	fi
}

ensure_isaac_system_build_exists

run_nav2_experience_batch() {
	local suite_name="$1"
	local batch_name="$2"
	echo "Starting test batch: ${batch_name} (suite: ${suite_name})"
	resim test-suites run --project "${PROJECT_NAME}" \
		--test-suite "${suite_name}" \
		--batch-name "${batch_name}" \
		--build-id "${ISAAC_SIM_BUILD_ID}" \
		--pool-labels 'resim:metrics2:k8s' \
		--sync-metrics-config --metrics-config-path .resim/metrics/config.resim.yml \
		--allowable-failure-percent 25
}

# Warehouse suite first, then the larger hospital suite (same build + metrics sync flags).
run_nav2_experience_batch "${WAREHOUSE_DEMO_SUITE_NAME}" "Warehouse Demo @ ${COMMIT_SHA}"
run_nav2_experience_batch "${HOSPITAL_DEMO_SUITE_NAME}" "Hospital Demo @ ${COMMIT_SHA}"
# pause

# echo "Once batches complete, create reports (same API/auth as this run), for example:"
# printf 'export RESIM_URL=%q RESIM_AUTH_URL=%q\n' "${RESIM_URL}" "${RESIM_AUTH_URL}"
# echo "resim reports create --branch main --metrics-build-id ${DEFAULT_REPORT_METRICS_BUILD_ID} --project \"${PROJECT_NAME}\" --test-suite \"${WAREHOUSE_DEMO_SUITE_NAME}\" --length 1"
# echo "resim reports create --branch main --metrics-build-id ${DEFAULT_REPORT_METRICS_BUILD_ID} --project \"${PROJECT_NAME}\" --test-suite \"${HOSPITAL_DEMO_SUITE_NAME}\" --length 1"
