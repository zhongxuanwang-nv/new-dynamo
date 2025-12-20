#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

export BLOCK_SIZE=16
export MAX_NUM_BLOCKS=15

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# --enforce-eager is added for quick deployment. for production use, need to remove this flag
CUDA_VISIBLE_DEVICES=4 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --block-size $BLOCK_SIZE --num-gpu-blocks-override $MAX_NUM_BLOCKS &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
DYN_VLLM_KV_EVENT_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=5 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker \
    --block-size $BLOCK_SIZE --num-gpu-blocks-override $MAX_NUM_BLOCKS
