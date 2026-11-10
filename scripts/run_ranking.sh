#!/usr/bin/env bash
# Ranking accuracy (Table IV): Recall@15 and AP on Friedman1, Smooth and F1-F10
# (plus the null Additive design), n=10,000, seeds 0-31.
#   1. LENS, TreeSHAP (XGBoost) and LENS, Integrated Hessians, NID (softplus MLP)
#      on the same fitted models.
#   2. Friedman H^2 on the same XGBoost models and grids: Recall@15 from the
#      ranking run, AP from a replay that scores the full ranking.
#   3. Reference implementations (TreeSHAP-IQ, SHAP-IQ, Archipelago, Faith-Shap,
#      hstats, iRF) in their own environments; see README for the checkouts.
# Extra settings: EXTERNAL (official checkouts), SHAPIQ_PY, IRF_PY, RSCRIPT.
source "$(dirname "$0")/common.sh"
RUN=${RUN:-runs/ranking}
EXTERNAL=${EXTERNAL:-third_party/official_baselines}
SHAPIQ_PY=${SHAPIQ_PY:-$PY}
IRF_PY=${IRF_PY:-python}
RSCRIPT=${RSCRIPT:-Rscript}
DGPS="Additive Friedman1 Smooth F1 F2 F3 F4 F5 F6 F7 F8 F9 F10"
SEEDS=${SEEDS:-$(seq 0 31)}
DGP_CSV=$(echo $DGPS | tr ' ' ,); SEED_CSV=$(echo $SEEDS | tr ' ' ,)
mkdir -p "$RUN/logs"
tasks=$RUN/tasks.txt; : > "$tasks"
for d in $DGPS; do for s in $SEEDS; do
  for t in xgboost softplus_mlp; do
    out=$RUN/teacher_family/teacher=$t/dgp=$d/seed=$s
    [ -f "$out/DONE" ] || echo "$PY -m lens.experiments.teacher_family_sensitivity --teacher $t --dgp $d --seed $s --n 10000 --top-k 15 --grid-points 12 --grid-batch-size 144 --ranking-max-samples 512 --n-bootstrap 999 --alpha 0.05 --device cuda --mlp-max-epochs 200 --mlp-min-epochs 30 --mlp-patience 15 --ih-n-explain 64 --external-root $EXTERNAL --output $out >> $RUN/logs/tf_${t}_${d}_$s.log 2>&1"
  done
  out=$RUN/union/dgp=$d/seed=$s
  [ -f "$out/DONE" ] || echo "$PY -m lens.experiments.synthetic_union_end_to_end --dgp $d --seed $s --n 10000 --top-k 15 --n-permutations 500 --grid-points 12 --grid-batch-size 144 --ranking-max-samples 3000 --rankers LENS-raw,LENS-debiased,Friedman-H2,TreeSHAP,FAST,NID,Random --nid-hidden 128,64 --nid-epochs 200 --nid-lr 0.001 --alpha 0.05 --device cuda --cache-mode exact_statistics --output $out >> $RUN/logs/union_${d}_$s.log 2>&1"
  for m in Archipelago Faith-Shap NID-official Integrated-Hessians TreeSHAP-IQ SHAP-IQ; do
    py=$PY; [[ $m == *SHAP-IQ ]] && py=$SHAPIQ_PY
    out=$RUN/official/method=$m/dgp=$d/seed=$s
    [ -f "$out/DONE" ] || echo "OMP_NUM_THREADS=1 $py -m lens.experiments.official_stage1_baseline --method $m --dgp $d --seed $s --device cuda --external-root $EXTERNAL --output $out >> $RUN/logs/official_${m}_${d}_$s.log 2>&1"
  done
  out=$RUN/official_r/method=hstats/dgp=$d/seed=$s
  [ -f "$out/DONE" ] || echo "mkdir -p $out && $PY -m lens.experiments.official_r_baseline_prepare --method hstats --dgp $d --seed $s --output $out && OMP_NUM_THREADS=1 $RSCRIPT scripts/ranking/official_r_baseline_cell.R hstats $out >> $RUN/logs/hstats_${d}_$s.log 2>&1"
  out=$RUN/official_irf/dgp=$d/seed=$s
  [ -f "$out/DONE" ] || echo "mkdir -p $out && $PY -m lens.experiments.official_irf_prepare --dgp $d --seed $s --output $out && OMP_NUM_THREADS=1 $IRF_PY scripts/ranking/official_irf_cell.py --input $out/input.npz --official-root $EXTERNAL/irf --output $out >> $RUN/logs/irf_${d}_$s.log 2>&1"
done; done >> "$tasks"
run_tasks "$tasks"

# H^2 AP replay on the union run's fitted models.
for k in $(seq 0 $((${#GPU_LIST[@]} - 1))); do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$k]} $PY -m lens.experiments.friedman_h2_full_ap_replay --output "$RUN/h2_replay" \
    --source "$RUN/union" --device cuda --n-shards ${#GPU_LIST[@]} --shard-index $k &
done
wait

# Aggregates (the files under results/ranking/).
A=$RUN/aggregates; mkdir -p "$A"
$PY scripts/ranking/aggregate_teacher_family_stage1_subset.py --root "$RUN/teacher_family" \
  --output "$A/teacher_family_stage1_aggregate.json" --manifest "$A/teacher_family_manifest.txt"
$PY -m lens.experiments.stage1_reference_aggregate --root "$RUN/union" --dgps "$DGP_CSV" --seeds "$SEED_CSV" \
  --output "$A/stage1_reference_aggregate.json"
$PY -m lens.experiments.friedman_h2_full_ap_replay --output "$RUN/h2_replay" --source "$RUN/union" --aggregate
cp "$RUN/h2_replay/aggregate.json" "$A/friedman_h2_full_ap_replay_aggregate.json"
$PY -m lens.experiments.official_stage1_aggregate --root "$RUN/official" \
  --methods Archipelago,Faith-Shap,NID-official,Integrated-Hessians,TreeSHAP-IQ,SHAP-IQ --dgps "$DGP_CSV" \
  --seeds "$SEED_CSV" --output "$A/official_stage1_six_method_aggregate.json"
$PY -m lens.experiments.official_supplemental_aggregate --kind r --root "$RUN/official_r" --dgps "$DGP_CSV" --seeds "$SEED_CSV" \
  --output "$A/official_r_baselines_aggregate.json"
$PY -m lens.experiments.official_supplemental_aggregate --kind irf --root "$RUN/official_irf" --dgps "$DGP_CSV" --seeds "$SEED_CSV" \
  --output "$A/official_irf_aggregate.json"
