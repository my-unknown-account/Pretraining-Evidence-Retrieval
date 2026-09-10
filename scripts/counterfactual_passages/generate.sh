#!/bin/bash
set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set."
  exit 1
fi

python3 generate.py \
  --input ../../data/exposure_data.json \
  --output ../../data/contradictions.json \
  --wikidata-cache ../../data/wikidata_truth_cache.json \
  --gen-model gpt-5.6-luna \
  --verify-model gpt-5.6-terra \
  --max-candidates 200 \
  --no-web-verify \
  --resume
