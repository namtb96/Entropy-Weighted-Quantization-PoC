#!/bin/bash

# Auto-generated quantization command based on EWQ plan.
# Plan Hash: de636fa5ca864c2a

./build/bin/llama-quantize \
  --allow-requantize \
  --imatrix ./namtb/Qwen3-8B-imatrix.dat \
  --tensor-type "output.weight=f16" \
  --tensor-type "token_embd.weight=f16" \
  --tensor-type "blk.0.*=f16" \
  --tensor-type "blk.1.*=Q4_K" \
  --tensor-type "blk.2.*=Q4_K" \
  --tensor-type "blk.3.*=Q4_K" \
  --tensor-type "blk.4.*=f16" \
  --tensor-type "blk.5.*=Q8_0" \
  --tensor-type "blk.6.*=Q8_0" \
  --tensor-type "blk.7.*=Q8_0" \
  --tensor-type "blk.8.*=Q8_0" \
  --tensor-type "blk.9.*=Q8_0" \
  --tensor-type "blk.10.*=Q4_K" \
  --tensor-type "blk.11.*=Q8_0" \
  --tensor-type "blk.12.*=Q4_K" \
  --tensor-type "blk.13.*=Q8_0" \
  --tensor-type "blk.14.*=Q8_0" \
  --tensor-type "blk.15.*=Q8_0" \
  --tensor-type "blk.16.*=Q4_K" \
  --tensor-type "blk.17.*=Q8_0" \
  --tensor-type "blk.18.*=Q8_0" \
  --tensor-type "blk.19.*=Q4_K" \
  --tensor-type "blk.20.*=f16" \
  --tensor-type "blk.21.*=Q8_0" \
  --tensor-type "blk.22.*=Q4_K" \
  --tensor-type "blk.23.*=f16" \
  --tensor-type "blk.24.*=f16" \
  --tensor-type "blk.25.*=f16" \
  --tensor-type "blk.26.*=Q8_0" \
  --tensor-type "blk.27.*=f16" \
  --tensor-type "blk.28.*=Q8_0" \
  --tensor-type "blk.29.*=f16" \
  --tensor-type "blk.30.*=Q8_0" \
  --tensor-type "blk.31.*=f16" \
  --tensor-type "blk.32.*=Q8_0" \
  --tensor-type "blk.33.*=Q8_0" \
  --tensor-type "blk.34.*=Q8_0" \
  --tensor-type "blk.35.*=Q4_K" \
  ./models/Qwen3-8B-F16.gguf \
  ./models/Qwen3-8B-gguf-ewq.gguf \
  Q8_0 \
  8