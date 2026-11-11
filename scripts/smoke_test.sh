#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/finetuning-regression-tester-uv-cache}"
uv run --no-sync finetuning-regression-tester doctor
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
uv run --no-sync finetuning-regression-tester data generate --output "$tmpdir/data" --train 20 --valid 5 --test 8
uv run --no-sync finetuning-regression-tester data validate --dataset "$tmpdir/data"
uv run --no-sync finetuning-regression-tester demo --output "$tmpdir/demo"
uv run --no-sync finetuning-regression-tester report --comparison "$tmpdir/demo/comparison.json" --output "$tmpdir/export"
