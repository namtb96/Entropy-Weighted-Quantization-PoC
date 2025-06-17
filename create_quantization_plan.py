#!/usr/bin/env python3
"""
EWQ Plan Generator (CPU-based)

Mục đích duy nhất của script này là tạo ra file kế hoạch lượng tử hóa.
Toàn bộ quá trình tải model và phân tích entropy sẽ được thực hiện trên CPU
để không chiếm dụng VRAM.
"""
import torch
import torch.nn as nn
import numpy as np
import gc
import os
import json
import hashlib
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import warnings
from benchmark.utils import get_model_hash, get_plan_path

warnings.filterwarnings("ignore")

# === Cấu hình (Không thay đổi) ===
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_CACHE_DIR = "./models"
QUANTIZED_MODEL_CACHE_DIR = "./quantized_models"
ENTROPY_THRESHOLD_FACTOR = 0.65 # Ngưỡng entropy để quyết định lượng tử hóa
BATCH_SIZE = 16

def calculate_layer_entropy_efficient(layer_weights: torch.Tensor) -> float:
    weights_flat = layer_weights.detach().cpu().half().flatten().numpy()
    hist, _ = np.histogram(weights_flat, bins=128, density=True)
    hist = hist[hist > 0]; hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist)); del weights_flat
    return float(entropy)

def calculate_block_entropy_efficient(block: nn.Module) -> float:
    total_weighted_entropy, total_parameters = 0.0, 0
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight; num_params = weight.numel()
            total_weighted_entropy += calculate_layer_entropy_efficient(weight) * num_params
            total_parameters += num_params; del weight
    return total_weighted_entropy / total_parameters if total_parameters > 0 else 0.0

def create_quantization_plan_batched(model: nn.Module) -> Dict[int, str]:
    print("🔍 Creating quantization plan on CPU...")
    block_entropies = []
    num_blocks = len(model.model.layers)
    for i in range(num_blocks):
        print(f"  📊 Analyzing Block {i+1}/{num_blocks}...")
        block_entropy = calculate_block_entropy_efficient(model.model.layers[i])
        block_entropies.append(block_entropy)
        print(f"    ✅ Block {i} average entropy = {block_entropy:.6f}")
        gc.collect()
    entropies_array = np.array(block_entropies)
    mean_entropy, std_entropy = np.mean(entropies_array), np.std(entropies_array)
    threshold = mean_entropy - ENTROPY_THRESHOLD_FACTOR * std_entropy
    plan, counts = {}, {"raw": 0, "8-bit": 0, "4-bit": 0}
    for i, entropy in enumerate(block_entropies):
        decision = "raw" if entropy >= mean_entropy else "8-bit" if entropy >= threshold else "4-bit"
        plan[i], counts[decision] = decision, counts[decision] + 1
    print("\n🎯 Final Quantization Plan:")
    print(f"  - Raw Precision:  {counts['raw']} blocks\n  - 8-bit Quantize: {counts['8-bit']} blocks\n  - 4-bit Quantize: {counts['4-bit']} blocks")
    return plan

def main():
    print("🚀 EWQ Plan Generator (CPU-based)")
    print("=" * 60)
    
    model_config = {
        'base_model': MODEL_ID, 'entropy_factor': ENTROPY_THRESHOLD_FACTOR,
        'quant_method': 'ewq-bitsandbytes'
    }
    model_hash = get_model_hash(MODEL_ID, model_config)
    plan_path = get_plan_path(model_hash)

    if plan_path.exists():
        print(f"✅ Plan already exists: {plan_path}. Nothing to do.")
        return

    print(f"🔑 Model ID: {MODEL_ID}\n🔑 Config Hash: {model_hash}")
    
    # <<< THAY ĐỔI QUAN TRỌNG: Tải model lên CPU >>>
    print("\n📥 Loading base model to CPU (this will use system RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16, # Dùng float16 để tiết kiệm RAM
        device_map="cpu",         # Buộc model phải nằm trên CPU
        trust_remote_code=True,
    )
    
    print("✅ Base model loaded to CPU.")
    
    quant_plan = create_quantization_plan_batched(model)
    
    print(f"\n💾 Saving quantization plan to: {plan_path}")
    with open(plan_path, 'w') as f:
        json.dump(quant_plan, f, indent=2)

    print("\n🎊 Plan generation complete! You can now run the benchmark script.")

if __name__ == "__main__":
    main()