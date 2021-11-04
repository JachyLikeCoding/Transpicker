#!/usr/bin/env bash
set -x

EXP_DIR=exps/r50_deformable_detr_cryo_0623
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}\
    --coco_path coco_split \
    --lr 0.0002 \
    --num_queries 300 \
    --batch_size 2 \
    --enc_layers 3 \
    --dec_layers 3