#!/bin/bash

usage() {
    echo "Usage: $0 --dataset <dataset_name>"
    echo ""
    echo "Options:"
    echo "  --dataset, -d    Dataset name (default: omomo)"
    echo ""
    echo "Example:"
    echo "  $0 --dataset behave"
    exit 1
}

DATASET="omomo"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d) DATASET="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

export PYTHONPATH="$(pwd):$PYTHONPATH"

FULL_CMD="CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 train/train.py"
FULL_CMD+=" --debug 0"
FULL_CMD+=" --save_dir save/$DATASET"
FULL_CMD+=" --dataset $DATASET"
FULL_CMD+=" --use_hand_scalar_rot 1"
FULL_CMD+=" --uniform_weight 1"
FULL_CMD+=" --obj_trans_w 1"
FULL_CMD+=" --obj_w 0.6"
FULL_CMD+=" --contact_weight 0.3"
FULL_CMD+=" --hw 1"
FULL_CMD+=" --body_w 1"
FULL_CMD+=" --foot_weight 0.8"
FULL_CMD+=" --uniform_reg 1"
FULL_CMD+=" --cw 0"
FULL_CMD+=" --normalize 1"
FULL_CMD+=" --diffusion_steps 500"
FULL_CMD+=" --snr 0"
FULL_CMD+=" --lr 1e-4"
FULL_CMD+=" --batch_size 64"
FULL_CMD+=" --latent_dim 512"
FULL_CMD+=" --lr_anneal_steps 2500"
FULL_CMD+=" --dropout 0.1"
FULL_CMD+=" --model_type mdm"
FULL_CMD+=" --crop_len 300"
FULL_CMD+=" --arch trans_dec"
FULL_CMD+=" --text_encoder_type bert"
FULL_CMD+=" --mask_frames"
FULL_CMD+=" --use_ema"

echo "Running: $FULL_CMD"
echo "---"
eval "$FULL_CMD"