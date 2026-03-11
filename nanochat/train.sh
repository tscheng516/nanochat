# !/bin/bash

# Config
DEPTH=12
NORM_POS="peri" # Options: "pre", "reordered", "peri"/"sandwich", "post", "hybrid0".
# TM_NORM="qk"

# Paths 
ROOT="$(pwd)" 
KEYS_DIR="$ROOT/../.keys"
export WANDB_API_KEY="$(cat "$KEYS_DIR/wandb_api_key")"
export WANDB_RUN="d${DEPTH}_${NORM_POS}"

# Environment setup
eval "$(conda shell.bash hook)"
conda activate nanochat
export CUDA_VISIBLE_DEVICES="2,3,4,5"          # CHANGE THIS for different runs
# taskset -c "$(cat /sys/devices/system/node/node1/cpulist)" # NUMA node1 CPU(s):  16-31,144-159


export OMP_NUM_THREADS=1



torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=$DEPTH \
    --norm-pos=$NORM_POS \
    --target-param-data-ratio=10 \
    --device-batch-size=16 \
    --window-pattern=L \
    --run=$WANDB_RUN