#!/usr/bin/env bash
set -euo pipefail
export GRBC_BACKBONE=xvla
exec bash "$(dirname "$0")/train_common.sh" "$@"
