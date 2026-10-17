#!/usr/bin/env bash
set -euo pipefail
export GRBC_BACKBONE=pi05
exec bash "$(dirname "$0")/train_common.sh" "$@"
