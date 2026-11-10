#!/usr/bin/env bash
# Why H^2 ranks worse than energy (Section VI-D, Fig. 7): Delta, H^2 and the
# main-effect energy M of every pair on F1-F10.  CPU only.
source "$(dirname "$0")/common.sh"
RUN=${RUN:-runs/h2_diagnostic}
SHARDS=${SHARDS:-$WORKERS}
for k in $(seq 0 $((SHARDS - 1))); do
  $PY -m lens.experiments.h2_denominator_diagnostic --output "$RUN" --contexts 1000 --n-shards "$SHARDS" --shard-index "$k" &
done
wait
$PY -m lens.experiments.h2_denominator_diagnostic_summary --output "$RUN"
