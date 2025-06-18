#!/usr/bin/env python3
"""
EWQ Plan Generator (CPU-based) - Improved Algorithm

This script generates a quantization plan using an improved entropy analysis algorithm.
The entire process of loading the model and analyzing entropy is performed on the CPU
to avoid VRAM consumption.
"""
import torch
import torch.nn as nn
import numpy as np
import gc
import os
import json
import hashlib
from pathlib import Path
from transformers import AutoModelForCausalLM
from typing import Dict, Tuple, List
from benchmark.utils import get_model_hash, get_plan_path

# === Cấu hình (Có thể thay đổi) ===
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_CACHE_DIR = "./models"
QUANTIZED_MODEL_CACHE_DIR = "./quantized_models"
ENTROPY_THRESHOLD_FACTOR = 0.65 # Ngưỡng entropy để quyết định lượng tử hóa


def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj

# === CÁC HÀM TÍNH TOÁN ENTROPY ĐÃ ĐƯỢC NÂNG CẤP (TỪ SCRIPT MỚI) ===

def calculate_layer_entropy_improved(layer_weights: torch.Tensor, bins: int = 256) -> Tuple[float, Dict]:
    """
    Calculates detailed entropy and other statistical metrics for a layer's weights.
    """
    weights_flat = layer_weights.detach().cpu().float().flatten().numpy()
    
    hist, _ = np.histogram(weights_flat, bins=bins, density=True)
    hist = hist[hist > 0]; hist = hist / np.sum(hist)
    
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    metrics = {
        'std': np.std(weights_flat),
        'mean': np.mean(weights_flat),
        'range': np.ptp(weights_flat),
        'sparsity': np.sum(np.abs(weights_flat) < 1e-6) / len(weights_flat),
        'effective_bits': shannon_entropy
    }
    
    del weights_flat
    return float(shannon_entropy), metrics

def calculate_block_entropy_improved(block: nn.Module) -> Dict:
    """
    Calculates the weighted average entropy and gathers detailed info for a block.
    """
    layer_entropies, layer_weights, layer_info, total_params = [], [], {}, 0
    
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            num_params = weight.numel()
            entropy, metrics = calculate_layer_entropy_improved(weight)
            
            layer_entropies.append(entropy)
            layer_weights.append(num_params)
            layer_info[name] = {'entropy': entropy, 'params': num_params, 'metrics': metrics}
            total_params += num_params
            del weight
    
    if total_params == 0:
        return {'weighted_entropy': 0.0, 'layer_info': {}, 'total_params': 0}
    
    weighted_entropy = np.average(layer_entropies, weights=layer_weights)
    
    return {
        'weighted_entropy': float(weighted_entropy),
        'layer_info': layer_info,
        'total_params': int(total_params)
    }

def analyze_entropy_distribution(block_results: List[Dict]) -> Dict:
    """
    Analyzes the distribution of entropies across all model blocks.
    """
    entropies = [result['weighted_entropy'] for result in block_results]
    entropies_array = np.array(entropies)
    
    return {
        'mean': np.mean(entropies_array),
        'std': np.std(entropies_array),
        'median': np.median(entropies_array),
        'q25': np.percentile(entropies_array, 25),
        'q75': np.percentile(entropies_array, 75)
    }

def create_quantization_plan_improved(model: nn.Module, entropy_factor: float) -> Tuple[Dict[int, str], Dict]:
    """
    Creates a quantization plan using the improved entropy analysis.
    """
    print("🔍 Creating quantization plan using improved algorithm...")
    
    block_results = []
    num_blocks = len(model.model.layers)
    
    for i in range(num_blocks):
        print(f"  📊 Analyzing Block {i+1}/{num_blocks}...")
        block_result = calculate_block_entropy_improved(model.model.layers[i])
        block_results.append(block_result)
        print(f"    ✅ Block {i} weighted entropy = {block_result['weighted_entropy']:.6f}")
        gc.collect()
    
    distribution = analyze_entropy_distribution(block_results)
    print(f"\n📈 Entropy Distribution Analysis:")
    print(f"  - Mean: {distribution['mean']:.6f}, Std: {distribution['std']:.6f}")
    print(f"  - Q25: {distribution['q25']:.6f}, Q75: {distribution['q75']:.6f}")

    # Sử dụng chiến lược "adaptive" từ script mới của bạn làm mặc định
    param_weights = [result['total_params'] for result in block_results]
    weighted_mean = np.average([res['weighted_entropy'] for res in block_results], weights=param_weights)
    threshold = weighted_mean - entropy_factor * distribution['std']
    
    print(f"\n🎯 Calculated Threshold (Adaptive Strategy): {threshold:.6f}")
    
    plan, counts = {}, {"raw": 0, "8-bit": 0, "4-bit": 0}
    
    for i, result in enumerate(block_results):
        entropy = result['weighted_entropy']
        
        # Logic ra quyết định tinh vi hơn từ script mới của bạn
        if entropy >= distribution['q75']:  # High entropy -> keep precision
            decision = "raw"
        elif entropy >= threshold:         # Medium entropy -> moderate quantization
            decision = "8-bit"
        else:                              # Low entropy -> aggressive quantization
            decision = "4-bit"
            
        plan[i] = decision
        counts[decision] += 1
    
    print("\n🎯 Final Quantization Plan:")
    print(f"  - Raw Precision:  {counts['raw']} blocks")
    print(f"  - 8-bit Quantize: {counts['8-bit']} blocks")
    print(f"  - 4-bit Quantize: {counts['4-bit']} blocks")
    
    # Gói kết quả phân tích để lưu lại
    analysis = {
        'block_results': block_results,
        'distribution': distribution,
        'selected_threshold': threshold,
        'plan_statistics': counts
    }
    
    return plan, analysis

def main():
    print("🚀 EWQ Plan Generator (CPU-based, Improved Algorithm)")
    print("=" * 60)
    
    model_config = {
        'base_model': MODEL_ID, 'entropy_factor': ENTROPY_THRESHOLD_FACTOR,
        'quant_method': 'ewq-bitsandbytes' # Hoặc một định danh phù hợp
    }
    model_hash = get_model_hash(MODEL_ID, model_config)
    plan_path = get_plan_path(model_hash)

    if plan_path.exists():
        print(f"✅ Plan already exists: {plan_path}. Nothing to do.")
        return

    print(f"🔑 Model ID: {MODEL_ID}\n🔑 Config Hash: {model_hash}")
    
    # Tải model lên CPU
    print("\n📥 Loading base model to CPU (this may use a lot of system RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("✅ Base model loaded to CPU.")
    
    # Tạo plan bằng thuật toán mới
    plan, analysis = create_quantization_plan_improved(model, ENTROPY_THRESHOLD_FACTOR)
    
    # Lưu cả plan và phân tích chi tiết
    save_data = {
        'plan': plan
    }

    print(f"\n💾 Saving quantization plan and analysis to: {plan_path}")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    command_str = generate_llama_quantize_command(plan)
    command_path = plan_path.with_name(f"quantize_command_{model_hash}.sh")
    
    print(f"💾 Saving quantization command script to: {command_path}")
    with open(command_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Auto-generated quantization command based on EWQ plan.\n")
        f.write(f"# Plan Hash: {model_hash}\n\n")
        f.write(command_str)

    print("\n🎊 Plan generation complete! You can now run the main benchmark script.")

    del model
    gc.collect()

def generate_llama_quantize_command(plan: Dict[int, str]) -> str:
    """
    Generates the llama-quantize command from a plan with hardcoded file paths.
    """
    tensor_types_str = []
    
    # Thêm các tensor cố định
    tensor_types_str.append('--tensor-type "output.weight=f16"')
    tensor_types_str.append('--tensor-type "token_embd.weight=f16"')
    
    # Tạo các dòng tensor type từ plan
    for block_idx, decision in sorted(plan.items()):
        quant_type = "f16" # Mặc định
        if decision == "8-bit":
            quant_type = "Q8_0"
        elif decision == "4-bit":
            quant_type = "Q4_K"
        tensor_types_str.append(f'--tensor-type "blk.{block_idx}.*={quant_type}"')

    # Nối các dòng lại với nhau, mỗi dòng thụt vào và có dấu \
    formatted_tensor_types = " \\\n  ".join(tensor_types_str)

    # Dựng câu lệnh hoàn chỉnh với các đường dẫn cố định
    command = f"""./build/bin/llama-quantize \\
  --allow-requantize \\
  --imatrix ./namtb/Qwen3-8B-imatrix.dat \\
  {formatted_tensor_types} \\
  ./models/Qwen3-8B-F16.gguf \\
  ./models/Qwen3-8B-gguf-ewq.gguf \\
  Q8_0 \\
  8"""
    
    return command

if __name__ == "__main__":
    main()