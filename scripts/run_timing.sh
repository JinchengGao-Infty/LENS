#!/usr/bin/env bash
# Wall-clock cost (Fig. 6): one cell per GPU at a time, seed 3.
source "$(dirname "$0")/common.sh"
RUN=${RUN:-runs/timing}
DATA=${DATA:-runs/synthetic_data}   # written by run_synthetic.sh
WORKERS=${#GPU_LIST[@]}
mkdir -p "$RUN/logs"
tasks=$RUN/tasks.txt; : > "$tasks"
add() {  # dataset p method
  local d=$1 p=$2 m=$3 out=$RUN/${1}_p${2}_seed3_${3}.json data model
  case $m in
    diamond|diamond_matched|diamond_mlp)
      model=${m#diamond}; model=${model#_}; [ -z "$model" ] && model=xgboost; [ "$model" = matched ] && model=xgboost_matched
      if [[ $d == F* ]]; then data=$DATA/${d}_seed3.npz; else data=$DATA/${d}_p${p}_seed3.npz; fi
      echo "$DPY lens/experiments/icde_diamond_adapter.py run --diamond-src $DIAMOND_SRC --data $data --seed 3 --model $model --output $out >> $RUN/logs/$d.log 2>&1";;
    *) echo "$PY -m lens.experiments.icde_main_comparison --dataset $d --p $p --seed 3 --data-dir $DATA --method $m --output $out >> $RUN/logs/$d.log 2>&1";;
  esac >> "$tasks"
}
for p in 50 30 10; do for m in iloco h2_refit iloco_mp lens diamond diamond_matched fast; do add sparse_pairs $p $m; done; done
for m in lens_mlp diamond_mlp; do add F1 30 $m; done
run_tasks "$tasks"
