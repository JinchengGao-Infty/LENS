# Shared settings, sourced by the run scripts.  Override any of them from the shell.
#   PY           python of the main environment (requirements.txt)
#   DPY          python of the DIAMOND environment (requirements-diamond.txt)
#   DIAMOND_SRC  the src/ directory of https://github.com/batmen-lab/diamond
#   GPUS         space-separated GPU indices to use
#   WORKERS      number of cells run at the same time (several per GPU is fine)
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PY:-python}
DPY=${DPY:-python}
DIAMOND_SRC=${DIAMOND_SRC:-third_party/diamond/src}
GPUS=${GPUS:-0}
WORKERS=${WORKERS:-4}
export PYTHONPATH="$ROOT:$ROOT/lens_method${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
read -r -a GPU_LIST <<< "$GPUS"

# run_tasks FILE: run each line of FILE as a shell command, WORKERS at a time;
# line k gets CUDA_VISIBLE_DEVICES set to the (k mod #GPUs)-th GPU.
run_tasks() {
  local n=${#GPU_LIST[@]}
  awk -v n="$n" -v gpus="$GPUS" 'BEGIN{split(gpus,g," ")} {print "CUDA_VISIBLE_DEVICES=" g[(NR-1)%n+1] " " $0}' "$1" \
    | xargs -P "$WORKERS" -I{} bash -c '{}'
}
