#!/usr/bin/env bash
# Real data (Section VII): freMTPL2 claim frequency at three training sizes, HELOC
# and UCI credit default, splits 30-34.  Each split file holds every list of the
# ablation (LENS, energy top-|LENS|, single model, both models over the floor,
# point estimate over the boundary without the test, all pairs).
source "$(dirname "$0")/common.sh"
RUN=${RUN:-runs/real}
SPLITS=${SPLITS:-30 31 32 33 34}
mkdir -p "$RUN/logs"
tasks=$RUN/tasks.txt; : > "$tasks"
for s in $SPLITS; do
  for size in full 50000 20000; do
    if [ $size = full ]; then out=$RUN/fremtpl2/full; extra=""; else out=$RUN/fremtpl2/train$size; extra="--train-size $size"; fi
    [ -s $out/split$s.json ] || echo "$PY -m lens.experiments.fremtpl2_poisson_allpair --split-seed $s $extra --output $out >> $RUN/logs/fremtpl2_${size}_$s.log 2>&1" >> "$tasks"
  done
  for ds in heloc uci_credit; do
    out=$RUN/credit/$ds
    [ -s $out/split$s.json ] || echo "$PY -m lens.experiments.credit_logistic_allpair --dataset $ds --data-dir data --split-seed $s --output $out >> $RUN/logs/${ds}_$s.log 2>&1" >> "$tasks"
  done
done
run_tasks "$tasks"
# The freMTPL2 tables drawn in Fig. 1 and Fig. 8 (split 30).
$PY -m lens.experiments.fremtpl2_overview_tables --split-seed 30 --output "$RUN/overview_tables/split30.json"
