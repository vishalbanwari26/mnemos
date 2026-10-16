#!/usr/bin/env bash
# Build the static GitHub Pages demo of the dashboard into dashboard/dist.
# Needs a recording first: see scripts/record_demo.py (writes dashboard/demo-data).
set -euo pipefail
cd "$(dirname "$0")/../dashboard"
test -d demo-data || { echo "dashboard/demo-data is missing: run scripts/record_demo.py first" >&2; exit 1; }
VITE_STATIC=1 VITE_BASE="${VITE_BASE:-/mnemos/}" npm run build
cp -R demo-data dist/demo-data
touch dist/.nojekyll
echo "built dashboard/dist ($(du -sh dist | cut -f1))"
