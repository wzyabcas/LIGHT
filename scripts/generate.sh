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

[[ -z "${MODEL_PATH[$DATASET]}" ]] && { echo "Error: unknown dataset '$DATASET'"; exit 1; }

FULL_CMD="python -m sample.generate"
FULL_CMD+=" --model_path ${MODEL_PATH[$DATASET]}"
FULL_CMD+=" --num_repetitions 1"
FULL_CMD+=" --batch_size 64"
FULL_CMD+=" --debug 1"

echo "Running: $FULL_CMD"
echo "---"
eval "$FULL_CMD"