#!/usr/bin/env bash
set -euo pipefail

# Topik 4 Bonus - jalankan di GPU03 / RTX 3080
# Output disimpan ke file teks agar bisa diberikan ke AI sebagai context laporan.

cd "$(dirname "$0")"

OUT="topik4_output_$(date +%Y%m%d_%H%M%S).txt"
JSON="topik4_results.json"

python3 topik4_gpu_ai_experiment.py \
  --samples 200000 \
  --features 128 \
  --classes 4 \
  --epochs 8 \
  --batch-size 2048 \
  --output-json "$JSON" | tee "$OUT"

echo
echo "Output teks: $OUT"
echo "Output JSON: $JSON"
