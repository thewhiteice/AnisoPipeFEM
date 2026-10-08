#!/bin/bash
set -e
cd "$(dirname "$0")/.."

export SCAN_GLOB="configs/**/*.yaml"
export SCAN_PARAM="d_limit"
export SCAN_VALUES="0.3,0.5,0.7,0.8,0.85,0.88,0.9,0.92,0.95,0.99"

python -m scripts.scan
