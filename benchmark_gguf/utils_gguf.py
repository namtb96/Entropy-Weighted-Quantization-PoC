import json
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

def get_model_hash(model_id: str, config: dict) -> str:
    """Tạo một hash duy nhất cho một cấu hình model GGUF cụ thể."""
    config_str = f"{model_id}-{json.dumps(config, sort_keys=True)}"
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def create_qwen_prompt(user_prompt):
    """Tạo prompt theo định dạng chat của Qwen. Giữ nguyên vì đây là đặc trưng của model."""
    system_prompt = "You are a helpful assistant. Provide a direct and concise answer to the user's request. Do not show your thought process, reasoning steps, or self-correction. Respond only with the final answer."
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def download_gguf_model(repo_id: str, filename: str, cache_dir: str) -> Path:
    """
    Tải file model GGUF từ Hugging Face Hub nếu chưa có và trả về đường dẫn.
    """
    print(f"  📥 Checking for GGUF model file: {filename}")
    model_path = Path(cache_dir) / filename
    if not model_path.exists():
        print(f"  ⏬ Model not found locally. Downloading from '{repo_id}'...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=cache_dir,
                local_dir_use_symlinks=False # Nên đặt là False để tránh vấn đề trên Windows
            )
            # hf_hub_download có thể đặt file vào một thư mục con snapshots, cần tìm đúng đường dẫn
            # Tuy nhiên, với cách dùng này, nó sẽ lưu trực tiếp vào cache_dir
            print(f"  ✅ Download complete. Model saved at: {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            print(f"  ❌ Failed to download model: {e}")
            raise
    else:
        print("  👍 Model found in cache.")
    return model_path