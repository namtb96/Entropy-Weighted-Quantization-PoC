#!/usr/bin/env python3
"""
EWQ Plan Generator (Tensor-Level, CPU-based) - Fixed Version

This script generates a quantization plan using tensor-level entropy analysis
for the most fine-grained optimization, including Linear, LayerNorm, and other tensors.
"""
import torch
import torch.nn as nn
import numpy as np
import gc
import os
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM
from typing import Dict, Tuple, List, Any
from benchmark.utils import get_model_hash, get_plan_path

# === Cấu hình (Có thể thay đổi) ===
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_CACHE_DIR = "./models"
ENTROPY_THRESHOLD_FACTOR = 0.55  # Điều chỉnh tinh tế

# TENSOR type importance weights for decision making (Chi tiết hơn)
TENSOR_IMPORTANCE = {
    'output.weight': 1.9,           # Tăng lên để đảm bảo f16
    'token_embd.weight': 1.9,       # Tăng lên để đảm bảo f16
    'attn_q.weight': 1.45,          # Tăng lên một chút
    'attn_k.weight': 1.35,          # Tăng lên một chút
    'attn_v.weight': 1.35,          # Tăng lên một chút
    'attn_output.weight': 1.25,     # Tăng lên một chút
    'ffn_gate.weight': 1.05,        # Tăng lên một chút
    'ffn_up.weight': 1.05,          # Tăng lên một chút
    'ffn_down.weight': 1.15,        # Tăng lên một chút
    'attn_norm.weight': 1.55,       # Tăng lên một chút
    'ffn_norm.weight': 1.55,        # Tăng lên một chút
    'tok_embeddings.weight': 1.9,   # Tăng lên để đảm bảo f16
    'output_norm.weight': 1.8,      # Tăng lên để đảm bảo f16
    # Mặc định
    '.weight': 1.0,
    '.bias': 1.25,                  # Tăng lên một chút
}

def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_json_serializable(i) for i in obj]
    return obj

def convert_hf_name_to_gguf(hf_name: str) -> str:
    """Converts a Hugging Face tensor name to a GGUF compatible name."""
    # Xóa prefix 'model.'
    name = hf_name.replace("model.", "")
    
    # Ánh xạ các layer đặc biệt
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name == "norm.weight":
        return "output_norm.weight"

    # Ánh xạ các block
    # ví dụ: layers.10.self_attn.q_proj.weight -> blk.10.attn_q.weight
    m = re.match(r"layers\.(\d+)\.(self_attn|mlp)\.(.+)\.(weight|bias)", name)
    if not m:
        # Xử lý Layernorm trong block
        m = re.match(r"layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.(weight|bias)", name)
        if not m:
            return name # Trả về tên gốc nếu không khớp
        
        block_idx, norm_type, tensor_type = m.groups()
        norm_map = {
            'input_layernorm': 'attn_norm',
            'post_attention_layernorm': 'ffn_norm'
        }
        return f"blk.{block_idx}.{norm_map[norm_type]}.{tensor_type}"

    block_idx, block_type, layer_name, tensor_type = m.groups()

    if block_type == "self_attn":
        layer_map = {
            'q_proj': 'attn_q',
            'k_proj': 'attn_k',
            'v_proj': 'attn_v',
            'o_proj': 'attn_output'
        }
        gguf_layer_name = layer_map.get(layer_name, layer_name)
    elif block_type == "mlp":
        layer_map = {
            'gate_proj': 'ffn_gate',
            'up_proj': 'ffn_up',
            'down_proj': 'ffn_down'
        }
        gguf_layer_name = layer_map.get(layer_name, layer_name)
    else:
        gguf_layer_name = layer_name

    return f"blk.{block_idx}.{gguf_layer_name}.{tensor_type}"

def get_tensor_importance(gguf_name: str) -> float:
    """Determines the importance weight of a tensor based on its GGUF name."""
    # Ưu tiên khớp chính xác nhất
    for pattern, weight in sorted(TENSOR_IMPORTANCE.items(), key=lambda x: len(x[0]), reverse=True):
        if pattern in gguf_name:
            return weight
    return 1.0  # Default importance

def calculate_tensor_entropy(tensor_weights: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
    """Calculates entropy and statistical metrics for a single tensor."""
    weights_flat = tensor_weights.detach().cpu().float().flatten().numpy()
    
    if weights_flat.size == 0:
        return 0.0, {}

    hist, _ = np.histogram(weights_flat, bins=256, density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    metrics = {
        'shannon_entropy': shannon_entropy,
        'std': np.std(weights_flat),
        'mean_abs': np.mean(np.abs(weights_flat)),
        'sparsity': np.sum(np.abs(weights_flat) < 1e-6) / len(weights_flat),
        'tensor_size': len(weights_flat)
    }
    
    del weights_flat
    return float(shannon_entropy), metrics

def analyze_tensor_by_tensor(model: nn.Module) -> List[Dict]:
    """Analyzes entropy for each individual tensor in the model."""
    print("🔬 Analyzing entropy tensor by tensor...")
    tensor_results = []
    
    # Iterate through all model parameters
    total_tensors = len(list(model.named_parameters()))
    print(f"  Total tensors to analyze: {total_tensors}")

    for i, (hf_name, tensor) in enumerate(model.named_parameters()):
        gguf_name = convert_hf_name_to_gguf(hf_name)
        if "lora" in gguf_name: # Bỏ qua LoRA adapters nếu có
            continue

        entropy, metrics = calculate_tensor_entropy(tensor)
        importance = get_tensor_importance(gguf_name)
        metrics['tensor_importance'] = importance
        
        tensor_result = {
            'hf_name': hf_name,
            'gguf_name': gguf_name,
            'entropy': entropy,
            'metrics': metrics
        }
        tensor_results.append(tensor_result)
        
        if (i+1) % 50 == 0 or (i+1) == total_tensors:
            print(f"  ✅ Processed {i+1}/{total_tensors}: {gguf_name} | entropy={entropy:.4f} | importance={importance:.2f}")

    print(f"\n📈 Analyzed {len(tensor_results)} tensors.")
    return tensor_results

def create_tensorwise_quantization_plan(tensor_results: List[Dict], entropy_factor: float) -> Tuple[Dict[str, str], Dict]:
    """Creates a detailed quantization plan for individual tensors."""
    print("🎯 Creating tensor-wise quantization plan...")
    
    # Phân tích phân phối entropy (có thể thêm chi tiết nếu cần)
    entropies = np.array([r['entropy'] for r in tensor_results if r['entropy'] > 0])
    importances = np.array([r['metrics']['tensor_importance'] for r in tensor_results if r['entropy'] > 0])
    
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    
    # Điều chỉnh ngưỡng để có phân phối cân bằng hơn
    high_precision_threshold = mean_entropy + entropy_factor * std_entropy
    medium_precision_threshold = mean_entropy - entropy_factor * std_entropy  # Thay đổi công thức
    
    print(f"\n🎯 Calculated Thresholds:")
    print(f"  - High Precision (f16): entropy >= {high_precision_threshold:.4f} (adjusted by importance)")
    print(f"  - Medium Precision (Q8_0): entropy >= {medium_precision_threshold:.4f} (adjusted by importance)")
    print(f"  - Low Precision (Q4_K): entropy < {medium_precision_threshold:.4f}")
    
    plan = {}
    counts = {"f16": 0, "Q8_0": 0, "Q4_K": 0, "COPY": 0} # COPY cho các tensor nhỏ
    
    # --- QUY TẮC CỨNG ---
    # Luôn giữ các tensor đặc biệt ở độ chính xác cao
    critical_tensors = ["output.weight", "token_embd.weight", "output_norm.weight"]
    for tensor_name in critical_tensors:
        plan[tensor_name] = "f16"
    
    for r in tensor_results:
        gguf_name = r['gguf_name']
        if gguf_name in plan: # Đã có quy tắc cứng
            counts[plan[gguf_name]] += 1
            continue

        entropy = r['entropy']
        importance = r['metrics']['tensor_importance']
        tensor_size = r['metrics']['tensor_size']
        
        # Điều chỉnh ngưỡng COPY - chỉ áp dụng cho tensor rất nhỏ
        if tensor_size < 128:  # Giảm từ 512 xuống 128
            decision = "COPY"
        else:
            # Điều chỉnh ngưỡng dựa trên độ quan trọng của tensor
            # Tensor càng quan trọng, ngưỡng để bị quantize xuống thấp càng cao
            adjusted_high = high_precision_threshold / max(importance, 1.0)  # Tránh chia cho 0
            adjusted_medium = medium_precision_threshold / max(importance, 1.0)
            
            # Thêm logic đặc biệt cho các tensor quan trọng
            # Điều chỉnh ngưỡng để có phân phối hợp lý: ~10% f16, ~35% Q8_0, ~55% Q4_K
            if importance >= 1.55 or entropy >= adjusted_high:  # Giảm xuống 1.55
                decision = "f16"
            elif importance >= 1.2 or entropy >= adjusted_medium:  # Tăng lên 1.2
                decision = "Q8_0"
            else:
                decision = "Q4_K"
        
        plan[gguf_name] = decision
        counts[decision] += 1
        
    print("\n🎯 Final Tensor-wise Quantization Plan:")
    print(f"  - f16 (High Precision): {counts['f16']} tensors")
    print(f"  - Q8_0 (Medium):        {counts['Q8_0']} tensors")
    print(f"  - Q4_K (Low):           {counts['Q4_K']} tensors")
    print(f"  - COPY (Untouched):     {counts['COPY']} tensors")
    print(f"  - Total tensors:        {sum(counts.values())}")
    
    # Thêm phân tích phần trăm
    total = sum(counts.values())
    print(f"\n📊 Distribution Percentages:")
    for quant_type, count in counts.items():
        percentage = (count / total) * 100
        print(f"  - {quant_type}: {percentage:.1f}%")
    
    analysis = {
        'plan_statistics': counts,
        'total_tensors': len(tensor_results)
    }
    
    return plan, analysis

def generate_tensorwise_llama_quantize_command(plan: Dict[str, str], model_id: str) -> str:
    """Generates the llama-quantize command from a tensor-wise plan - FIXED VERSION."""
    tensor_types_str = []
    
    # Kiểm tra và validate plan trước khi tạo command
    valid_quant_types = {"f16", "Q8_0", "Q4_K", "Q6_K", "Q5_0", "Q5_1", "COPY"}
    
    # Thêm các tensor vào command với validation
    for tensor_name, quant_type in sorted(plan.items()):
        if quant_type == "COPY": # Không cần chỉ định cho COPY
            continue
            
        # Validate quant_type
        if quant_type not in valid_quant_types:
            print(f"⚠️  Warning: Invalid quantization type '{quant_type}' for tensor '{tensor_name}', skipping...")
            continue
            
        # Validate tensor_name (không chứa ký tự đặc biệt có thể gây lỗi)
        if not tensor_name or '"' in tensor_name or '\n' in tensor_name:
            print(f"⚠️  Warning: Invalid tensor name '{tensor_name}', skipping...")
            continue
            
        tensor_types_str.append(f'--tensor-type "{tensor_name}={quant_type}"')
    
    # Kiểm tra nếu không có tensor nào hợp lệ
    if not tensor_types_str:
        print("⚠️  Warning: No valid tensor specifications found!")
        return ""
    
    formatted_tensor_types = " \\\n  ".join(tensor_types_str)
    model_name = model_id.split('/')[-1]

    # Đảm bảo thư mục output tồn tại
    command = f"""#!/bin/bash

# Auto-generated tensor-wise quantization command based on EWQ plan.
# Fixed version with proper error handling and validation

# Ensure output directory exists
mkdir -p ./models

# Check if input file exists
if [ ! -f "./models/{model_name}-F16.gguf" ]; then
    echo "❌ Error: Input file ./models/{model_name}-F16.gguf not found!"
    exit 1
fi

# Check if imatrix file exists
if [ ! -f "./namtb/{model_name}-imatrix.dat" ]; then
    echo "⚠️  Warning: imatrix file ./namtb/{model_name}-imatrix.dat not found!"
    echo "Continuing without imatrix..."
    IMATRIX_ARG=""
else
    IMATRIX_ARG="--imatrix ./namtb/{model_name}-imatrix.dat"
fi

# Run quantization with error handling
echo "🚀 Starting quantization..."
./build/bin/llama-quantize \\
  --allow-requantize \\
  $IMATRIX_ARG \\
  {formatted_tensor_types} \\
  ./models/{model_name}-F16.gguf \\
  ./models/{model_name}-gguf-ewq-tensorwise.gguf \\
  Q4_K \\
  8

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Quantization completed successfully!"
    echo "📁 Output: ./models/{model_name}-gguf-ewq-tensorwise.gguf"
else
    echo "❌ Quantization failed!"
    exit 1
fi"""
    
    return command

# Phần main với error handling tốt hơn
def main():
    print("🚀 EWQ Tensor-Level Plan Generator (CPU-based) - FIXED VERSION")
    print("=" * 70)
    
    try:
        model_config = {
            'base_model': MODEL_ID, 
            'entropy_factor': ENTROPY_THRESHOLD_FACTOR,
            'quant_method': 'ewq-tensorwise', # Đổi tên
            'granularity': 'tensor' # Đổi tên
        }
        model_hash = get_model_hash(MODEL_ID, model_config)
        plan_path = get_plan_path(model_hash)

        if plan_path.exists():
            print(f"✅ Plan already exists: {plan_path}. Nothing to do.")
            return

        print(f"🔑 Model ID: {MODEL_ID}\n🔑 Config Hash: {model_hash}")
        print(f"🎯 Granularity: Tensor-level (Optimal)")
        
        print("\n📥 Loading base model to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("✅ Base model loaded to CPU.")
        
        # Phân tích theo từng tensor
        tensor_results = analyze_tensor_by_tensor(model)
        
        # Tạo kế hoạch quantize theo tensor
        plan, analysis = create_tensorwise_quantization_plan(tensor_results, ENTROPY_THRESHOLD_FACTOR)
        
        # Lưu kế hoạch
        save_data = {
            'plan': plan,
            'analysis_summary': {
                'total_tensors': analysis['total_tensors'],
                'plan_statistics': analysis['plan_statistics'],
                'granularity': 'tensor-level'
            }
        }

        print(f"\n💾 Saving tensor-wise quantization plan to: {plan_path}")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, 'w') as f:
            json.dump(convert_to_json_serializable(save_data), f, indent=2)
        
        # Tạo command với error handling
        command_str = generate_tensorwise_llama_quantize_command(plan, MODEL_ID)
        if not command_str:
            print("❌ Failed to generate quantization command!")
            return
            
        command_path = plan_path.with_name(f"quantize_command_tensorwise_{model_hash}.sh")
        
        print(f"💾 Saving quantization command script to: {command_path}")
        with open(command_path, 'w') as f:
            f.write(command_str)
        
        os.chmod(command_path, 0o755)

        print("\n🎊 Tensor-wise plan generation complete!")
        print(f"📊 Generated plan for {analysis['total_tensors']} individual tensors")
        print("You can now run the quantization command with better error handling.")

        del model
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error during plan generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()