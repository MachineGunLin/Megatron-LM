#!/bin/bash
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="/home/infiniai/linrongjian/Megatron-LM/megatron/core/models/multimodal/output/demo_data_fix"
MODEL_NAME_OR_PATH="/home/infiniai/linrongjian/Megatron-LM/megatron/core/models/multimodal/output/init_model/init_0828"
# CHECKPOINT_PATH=/data/megrez-o-3b-lrj/finetune/output/init_model/init_0828
CHECKPOINT_PATH=/data/megrez-o-3b-lrj/finetune/ckpt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# --hidden-size 2560 \
# --num-attention-heads 16 \
# --max-position-embeddings 4096 \
# --encoder-seq-length 4096 \
# --model-max-length 4096 \
# --use-distributed-optimizer \
torchrun $DISTRIBUTED_ARGS \
    pretrain_megrezo.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --bf16 \
    --num-layers 1 \
    --hidden-size 1024 \
    --num-attention-heads 32 \
    --max-position-embeddings 1024 \
    --encoder-seq-length 1024 \
    --model-max-length 4096 \
    --tokenizer-type VlmTokenizer \
    --train-iters 20 \
    --lr 3e-4 \
    --lr-decay-style "cosine" \
    --min-lr 1e-5 \
    --loss-scale 65536 \
    --log-interval 1 \
    --weight-decay 0.1 \
    --adam-beta2 0.98 \
    --vlm-data-path $DATA_PATH \
    --lr-scheduler-type "cosine" \
    --model-name-or-path $MODEL_NAME_OR_PATH
    

# --load $CHECKPOINT_PATH \
# --no-load-optim
    