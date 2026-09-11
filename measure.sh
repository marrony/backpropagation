#!/bin/bash

cd distill
dev_bytes=$(uv run scripts/dump_samples.py data/seqkd_gold_dev.jsonl 2> /dev/null | LC_ALL=C tr -d '\000' | wc -c)
cd ..

dev_bytes=$(( $dev_bytes ))

make bin/bits_char 2>&1 > /dev/null

for V in 512 1024 2048 3072 5120; do
  make export "VOCAB_SIZE=${V}" 2>&1 > /dev/null
  ./bin/bits_char "--max-vocab=${V}" \
    "--dev-chars=${dev_bytes}" \
    "--train-ids=distill/cdata-${V}/train.ids.bin" \
    "--dev-ids=distill/cdata-${V}/dev.ids.bin" 2> /dev/null
  done

