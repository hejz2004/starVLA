#!/usr/bin/env bash

set -Eeuo pipefail

# main dir
cd /mnt/workspace/Native_Policy/user/hjz/starVLA

pip install -r requirements-ppu.txt

pip install -e .
