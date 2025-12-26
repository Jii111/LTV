#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

python run_zero.py --config_path configs/Aconfig_zero.py
