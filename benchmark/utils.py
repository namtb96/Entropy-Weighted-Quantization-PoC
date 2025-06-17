import json
import hashlib
from pathlib import Path

QUANTIZED_MODEL_CACHE_DIR = "./quantized_models"

def get_model_hash(model_id: str, config: dict) -> str:
    """Tạo một hash duy nhất cho một cấu hình model cụ thể."""
    config_str = f"{model_id}-{json.dumps(config, sort_keys=True)}"
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def get_plan_path(model_hash: str) -> Path:
    """Lấy đường dẫn đến file kế hoạch lượng tử hóa dựa trên hash."""
    cache_dir = Path(QUANTIZED_MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"quant_plan_{model_hash}.json"

def create_qwen_prompt(user_prompt):
    system_prompt = "You are a helpful assistant. Provide a direct and concise answer to the user's request. Do not show your thought process, reasoning steps, or self-correction. Respond only with the final answer."
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def get_object_debug_info(obj: object) -> dict:
    """
    Trích xuất các thuộc tính có thể serialize của một đối tượng để debug.
    
    Hàm này sẽ lặp qua các thuộc tính của đối tượng, bỏ qua các hàm và
    các thuộc tính private, sau đó chuyển đổi những gì có thể thành một dict.
    """
    debug_info = {}
    debug_info['__object_type__'] = str(type(obj))
    
    # Lấy danh sách tất cả các thuộc tính và phương thức
    for attr_name in dir(obj):
        # Bỏ qua các thuộc tính "magic" và "private" để output sạch hơn
        if attr_name.startswith('_'):
            continue
            
        try:
            value = getattr(obj, attr_name)
            
            # Bỏ qua các phương thức/hàm
            if callable(value):
                continue
            
            # Chỉ lấy các kiểu dữ liệu cơ bản có thể chuyển sang JSON
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                debug_info[attr_name] = value
            else:
                # Đối với các thuộc tính là đối tượng phức tạp khác, chỉ ghi lại kiểu của chúng
                debug_info[attr_name] = f"<Object of type: {type(value).__name__}>"

        except Exception as e:
            debug_info[attr_name] = f"<Error accessing attribute: {e}>"
            
    return debug_info