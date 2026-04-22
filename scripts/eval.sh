#!/bin/bash

usage() {
    echo "Usage: $0 --dataset <omomo|behave|interact>"
    echo ""
    echo "Options:"
    echo "  --dataset, -d    Dataset name: omomo, behave, interact"
    echo ""
    echo "Example:"
    echo "  $0 --dataset omomo"
    exit 1
}

DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d) DATASET="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" ]] && usage

declare -A MODEL_PATH=(
    [omomo]="save/omomo_pretrained/model000208000.pt"
    [behave]="save/behave_pretrained/model000288000.pt"
    [interact]="save/interact_pretrained/model000248000.pt"
)

declare -A DATASET_NAME=(
    [omomo]="omomo_correct"
    [behave]="behave_correct"
    [interact]="interact_correct"
)

[[ -z "${MODEL_PATH[$DATASET]}" ]] && { echo "Error: unknown dataset '$DATASET'"; exit 1; }

FULL_CMD="CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 eval/eval_t2hoi.py"
FULL_CMD+=" --model_path ${MODEL_PATH[$DATASET]}"
FULL_CMD+=" --dataset ${DATASET_NAME[$DATASET]}"

echo "Running: $FULL_CMD"
echo "---"
eval "$FULL_CMD"