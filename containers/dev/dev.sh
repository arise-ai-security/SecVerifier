#!/usr/bin/env bash
set -o pipefail

function get_docker() {
    echo "Docker is required to build and run OpenHands."
    echo "https://docs.docker.com/get-started/get-docker/"
    exit 1
}

function check_tools() {
	command -v docker &>/dev/null || get_docker
}

function exit_if_indocker() {
    if [ -f /.dockerenv ]; then
        echo "Running inside a Docker container. Exiting..."
        exit 1
    fi
}

##
# Determine repository root. Prefer git, but fall back to the script's parent tree
# so the script works when the files are extracted from a tarball or not in a git repo.
OPENHANDS_WORKSPACE=""
if OPENHANDS_WORKSPACE=$(git rev-parse --show-toplevel 2>/dev/null); then
    # got it from git
    :
else
    # Fall back to the script location: two levels up from containers/dev
    # Use BASH_SOURCE if available to get the script path even when sourced.
    SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd || true)
    if [ -n "$SCRIPT_DIR" ]; then
        # containers/dev is expected to live at <repo-root>/containers/dev
        # go two levels up from containers/dev to reach repo root
        OPENHANDS_WORKSPACE=$(cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd || true)
    fi
fi

if [ -z "$OPENHANDS_WORKSPACE" ]; then
    echo "Could not determine project root (not a git repo and script path unresolved)." >&2
    echo "Please run this script from the repository tree or install git." >&2
    exit 1
fi

cd "$OPENHANDS_WORKSPACE/containers/dev" || {
    echo "Failed to cd to '$OPENHANDS_WORKSPACE/containers/dev'" >&2
    exit 1
}
##
export BACKEND_HOST="0.0.0.0"
#
export SANDBOX_USER_ID=$(id -u)
export WORKSPACE_BASE=${WORKSPACE_BASE:-$OPENHANDS_WORKSPACE/workspace}

docker compose run --rm --service-ports "$@" dev

##
