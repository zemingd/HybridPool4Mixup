#!/usr/bin/env bash

set -e

source activate pyg

PROJECT=$PWD

# you can adapt these (to save on an external disk for example)
DATA=$PROJECT/data
SAVE=$PROJECT/saved_models

DATASET='Python800/Java250'
MODEL=$GNNs

LR=1e-3
CLIP=0.25
BATCHSIZE=80
RUNS=10
EPOCHS=100
LAY=5
PAT=20

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT

echo "PYTHONPATH: ${PYTHONPATH}"
echo started experiments

cd src



python main.py --gnn=$MODEL --drop_ratio=0 --lr=$LR  \
      --num_layer=$LAY  --emb_dim=300 --batch_size=$BATCHSIZE --runs=$RUNS --epochs=$EPOCHS --num_workers=$BATCHSIZE --dataset=$DATASET \
      --dir_data=$DATA  --dir_save=$SAVE --filename=$NAME --clip=$CLIP \
      --checkpointing=1 --checkpoint=$CHECKPOINT --patience=$PAT

echo "Run completed at:- "
date
