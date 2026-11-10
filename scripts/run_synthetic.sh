#!/usr/bin/env bash
# Synthetic comparison (Section VI): four designs x p in {10,30,50} and DIAMOND's
# F1-F10 at p=30, seeds 3-22, every method.  Writes runs/synthetic/*.json.
source "$(dirname "$0")/common.sh"
RUN=${RUN:-runs/synthetic}
DATA=${DATA:-runs/synthetic_data}
SEEDS=${SEEDS:-$(seq 3 22)}
mkdir -p "$RUN" "$DATA" "$RUN/logs"

# 1. Data: the four designs come from experiments/dgps.py, F1-F10 from DIAMOND's own generator.
tasks=$RUN/data_tasks.txt; : > "$tasks"
for d in sparse_pairs additive additive_corr pair_corr; do for p in 10 30 50; do for s in $SEEDS; do
  f=$DATA/${d}_p${p}_seed${s}.npz
  [ -s "$f" ] || echo "$PY -m lens.experiments.icde_main_comparison --dataset $d --p $p --seed $s --dump-data $f" >> "$tasks"
done; done; done
for i in $(seq 1 10); do for s in $SEEDS; do
  f=$DATA/F${i}_seed${s}.npz
  [ -s "$f" ] || echo "$DPY lens/experiments/icde_diamond_adapter.py dump-f --diamond-src $DIAMOND_SRC --data $f --seed $s --func $i" >> "$tasks"
done; done
run_tasks "$tasks"

# 2. One cell per (design, p, seed, method).
cell() {  # dataset p seed method
  local d=$1 p=$2 s=$3 m=$4 out=$RUN/${1}_p${2}_seed${3}_${4}.json data model
  [ -s "$out" ] && return
  case $m in
    diamond|diamond_matched|diamond_mlp)
      model=${m#diamond}; model=${model#_}; [ -z "$model" ] && model=xgboost; [ "$model" = matched ] && model=xgboost_matched
      if [[ $d == F* ]]; then data=$DATA/${d}_seed${s}.npz; else data=$DATA/${d}_p${p}_seed${s}.npz; fi
      echo "$DPY lens/experiments/icde_diamond_adapter.py run --diamond-src $DIAMOND_SRC --data $data --seed $s --model $model --output $out >> $RUN/logs/${d}_p${p}_seed${s}_${m}.log 2>&1";;
    *)
      echo "$PY -m lens.experiments.icde_main_comparison --dataset $d --p $p --seed $s --data-dir $DATA --method $m --output $out >> $RUN/logs/${d}_p${p}_seed${s}_${m}.log 2>&1";;
  esac
}
tasks=$RUN/tasks.txt; : > "$tasks"
for d in sparse_pairs additive additive_corr pair_corr; do for p in 50 30 10; do for s in $SEEDS; do
  for m in iloco h2_refit lens lens_mlp diamond diamond_matched diamond_mlp iloco_mp fast; do cell $d $p $s $m >> "$tasks"; done
done; done; done
for i in $(seq 1 10); do for s in $SEEDS; do
  for m in iloco h2_refit lens lens_mlp diamond diamond_matched diamond_mlp iloco_mp fast; do cell F$i 30 $s $m >> "$tasks"; done
done; done
run_tasks "$tasks"

# 3. Summaries: Tables III-V and the material-truth strengths of F1-F10.
$PY -m lens.experiments.icde_main_comparison_summary --run "$RUN" --data-dir "$DATA" > "$RUN/summary.txt"
$PY -m lens.experiments.icde_lens_floor_analysis --run "$RUN" > "$RUN/floor_analysis.txt"
$PY lens/experiments/icde_material_truth.py --run "$RUN" --data-dir "$DATA" \
  --gen-y "$DIAMOND_SRC/simulation/gen_Y.py" --output "$RUN/material_truth.json"
