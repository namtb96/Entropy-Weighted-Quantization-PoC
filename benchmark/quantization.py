import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes.nn as bnb
from typing import Dict, Tuple, Optional, Callable
import json
from huggingface_hub import hf_hub_download
import torch

from benchmark.utils import get_plan_path

def _find_and_replace(module: nn.Module, replacement_func: Callable, name_prefix=""):
    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, replacement_func(child, full_name))
        else:
            _find_and_replace(child, replacement_func, name_prefix=full_name)

def apply_quantization_balanced(model: nn.Module, plan: Dict[int, str]) -> nn.Module:
    print("  🔧 Applying balanced quantization plan on CPU...")
    for block_idx, quant_type in plan.items():
        if quant_type == "raw": continue
        block = model.model.layers[block_idx]
        block.to(torch.float16)
        def replacement_function(linear_module, module_name):
            if quant_type == "8-bit":
                q_layer = bnb.Linear8bitLt(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None, has_fp16_weights=False)
            elif quant_type == "4-bit":
                q_layer = bnb.Linear4bit(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None, compute_dtype=torch.float16, quant_type="nf4")
            else:
                return linear_module
            q_layer.weight.data.copy_(linear_module.weight.data)
            if linear_module.bias is not None:
                q_layer.bias.data.copy_(linear_module.bias.data)
            return q_layer
        _find_and_replace(block, replacement_function)
    print("  ✅ Quantization plan applied successfully on CPU model!")
    return model

def check_quantized_model_exists(model_hash: str) -> bool:
    return get_plan_path(model_hash).exists()

def load_quantized_model(model_id: str, model_hash: str, model_cache_dir: str) -> Tuple[Optional[nn.Module], Optional[AutoTokenizer]]:
    plan_path = get_plan_path(model_hash)
    print(f"  📄 Loading quantization plan from: {plan_path}")
    
    with open(plan_path, 'r') as f:
        data = json.load(f)
        plan_data = data.get('plan')
        quant_plan = {int(k): v for k, v in plan_data.items()}

    print("  📥 Loading base model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=model_cache_dir, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=model_cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    quantized_model = apply_quantization_balanced(model, quant_plan)
    
    print("  🚀 Deploying quantized model to GPU...")
    if torch.cuda.is_available():
        quantized_model.to("cuda")
    else:
        print("  ⚠️ WARNING: No CUDA device found. Benchmark will run on CPU.")

    return quantized_model, tokenizer

def load_original_model(model_id: str, model_cache_dir: str) -> Tuple[Optional[nn.Module], Optional[AutoTokenizer]]:
    """Tải model gốc với precision cao nhất có thể."""
    print("  📥 Loading original model to GPU...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ WARNING: No CUDA device found. Model will run on CPU.")
        # Trên CPU, float32 vẫn là lựa chọn hợp lý
        torch_dtype = torch.float32
        device_map = "cpu"
    else:
        # Luôn sử dụng bfloat16 trên GPU cho các model lớn.
        torch_dtype = torch.bfloat16
        device_map = "auto"
        print(f"  🎯 Using bfloat16 precision for optimal GPU performance.")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=model_cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True if torch.cuda.is_available() else False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=model_cache_dir,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("  ✅ Original model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"  ❌ Failed to load original model: {e}")
        return None, None