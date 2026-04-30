#!/bin/bash

usage() {
    echo "Usage: $0 --dataset <omomo|behave|interact> --mode <0-5> [extra args...]"
    echo ""
    echo "Options:"
    echo "  --dataset, -d    Dataset name: omomo, behave, interact"
    echo "  --mode, -m       Mode number: 0, 1, 2, 3, 4, 5"
    echo ""
    echo "Example:"
    echo "  $0 --dataset omomo --mode 2"
    exit 1
}

DATASET=""
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d) DATASET="$2"; shift 2 ;;
        --mode|-m)    MODE="$2";    shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" || -z "$MODE" ]] && usage
[[ "$MODE" =~ ^[0-5]$ ]] || { echo "Error: mode must be 0-5"; exit 1; }

# --- Model paths ---
declare -A MODEL_PATH=(
    [omomo]="save/omomo_pretrained/model000208000.pt"
    [behave]="save/behave_pretrained/model000288000.pt"
    [interact]="save/interact_pretrained/model000248000.pt"
)

[[ -z "${MODEL_PATH[$DATASET]}" ]] && { echo "Error: unknown dataset '$DATASET'"; exit 1; }

BASE_CMD="python -m sample.generate_guide --model_path ${MODEL_PATH[$DATASET]} --num_repetitions 1 --batch_size 64 --debug 1"

# --- Per-dataset, per-mode parameters ---
# Format: "df_delta df_weight df_divider df_upstop"
# (remaining params are constant per dataset or globally shared)

declare -A PARAMS

# omomo
PARAMS[omomo,0]="250 4.0 2 470"
PARAMS[omomo,1]="350 5.0 1 470"
PARAMS[omomo,2]="150 4.0 2 470"
PARAMS[omomo,3]="250 2.0 1 470"
PARAMS[omomo,4]="350 5.0 1 470"
PARAMS[omomo,5]="250 4.0 2 470"

# behave
PARAMS[behave,0]="200 5.0 1 470"
PARAMS[behave,1]="250 3.5 3 460"
PARAMS[behave,2]="200 5.0 1 470"
PARAMS[behave,3]="200 5.0 1 470"
PARAMS[behave,4]="200 5.0 1 470"
PARAMS[behave,5]="200 5.0 1 470"

# interact
PARAMS[interact,0]="100 2.5 3 470"
PARAMS[interact,1]="50  5.0 1 470"
PARAMS[interact,2]="75  5.0 1 470"
PARAMS[interact,3]="150 2.0 2 470"
PARAMS[interact,4]="350 5.0 1 470"
PARAMS[interact,5]="350 5.0 1 470"

P="${PARAMS[$DATASET,$MODE]}"
read -r DF_DELTA DF_WEIGHT DF_DIVIDER DF_UPSTOP <<< "$P"

FULL_CMD="$BASE_CMD"
FULL_CMD+=" --df_delta $DF_DELTA"
FULL_CMD+=" --df_weight $DF_WEIGHT"
FULL_CMD+=" --df_divider $DF_DIVIDER"
FULL_CMD+=" --df_upstop $DF_UPSTOP"
FULL_CMD+=" --df_decay 0"
FULL_CMD+=" --df_begin 0"
FULL_CMD+=" --df_cfg 1"
FULL_CMD+=" --df_mode $MODE"
FULL_CMD+=" --df_star 0"
FULL_CMD+=" --df_tweight 1.5"
FULL_CMD+=" --df_full_mode 0"
FULL_CMD+=" --df_eta 0.0"
FULL_CMD+=" --df_rescale 0.0"
FULL_CMD+=" --guidance_param 1.5"
FULL_CMD+=" --df_gw 0.0"
FULL_CMD+=" --df_add 0.0"
FULL_CMD+=" --df_r 0.0"
FULL_CMD+=" --df_mom 0.0"

echo "Running: $FULL_CMD"
echo "---"
eval "$FULL_CMD"