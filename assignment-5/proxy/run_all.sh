#!/bin/bash
# Run the three candidate mixtures through the tiny proxy, then summarise.
set -e
cd "$(dirname "$0")/.."
FILTER='Fatal|GIL|Thread|frame|Extension|Aborted|core dump'
MIXES="${MIXES:-naive_web ours code_heavy indic_first reasoning_fwd web_lean}"
for m in $MIXES; do
  echo "======== $m ========"
  THREADS=8 python3 -u proxy/train.py --name "$m" --steps 1500 --eval_every 300 \
    --block 128 --n_embd 192 --n_layer 4 --bs 16 2>&1 | grep -vE "$FILTER" || true
done
echo "======== summarize ========"
python3 -u proxy/summarize.py 2>&1 | grep -vE "$FILTER" || true
echo "ALL DONE"
