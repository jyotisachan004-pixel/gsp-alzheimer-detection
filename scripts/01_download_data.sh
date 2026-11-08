#!/usr/bin/env bash
# Download OpenNeuro ds004504 (Miltiadous et al., 2023) - public, no login needed.
# The paper used snapshot 1.0.6. If the anonymous mirror only serves the latest version, record the version in SETUP.md/NOTES.md.
set -euo pipefail
DEST=${1:-data/ds004504}
mkdir -p "$DEST" logs
pip install -q awscli
aws s3 sync --no-sign-request s3://openneuro.org/ds004504/ "$DEST" \
  --exclude "*" --include "participants.tsv" --include "dataset_description.json" \
  --include "derivatives/*" 2>&1 | tee logs/01_download.log
echo "Number of .set files:"; find "$DEST/derivatives" -name "*.set" | wc -l
