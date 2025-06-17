from llama_cpp import Llama
from pathlib import Path
from typing import Optional

from .utils_gguf import download_gguf_model

def load_gguf_model(
    repo_id: str,
    model_file: str,
    model_cache_dir: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 4096,
    verbose: bool = False
) -> Optional[Llama]:
    """
    Tải model GGUF sử dụng llama-cpp-python.

    Args:
        repo_id (str): ID của repository trên Hugging Face (ví dụ: 'Qwen/Qwen2-7B-Instruct-GGUF').
        model_file (str): Tên file .gguf cụ thể (ví dụ: 'qwen2-7b-instruct-q4_k_m.gguf').
        model_cache_dir (str): Thư mục để lưu trữ model.
        n_gpu_layers (int): Số lượng layer để offload lên GPU. -1 để offload tất cả.
        n_ctx (int): Kích thước ngữ cảnh tối đa của model.
        verbose (bool): Bật/tắt log chi tiết từ llama.cpp.

    Returns:
        Một đối tượng Llama hoặc None nếu có lỗi.
    """
    try:
        model_path = download_gguf_model(repo_id, model_file, model_cache_dir)
        
        print(f"  🚀 Loading GGUF model into memory...")
        print(f"     - Model Path: {model_path}")
        print(f"     - GPU Layers: {'All' if n_gpu_layers == -1 else n_gpu_layers}")
        print(f"     - Context Size: {n_ctx}")

        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=verbose,
            chat_format="chatml", # Qwen sử dụng định dạng ChatML
            logits_all=True  # <<< THAY ĐỔI QUAN TRỌNG: Bắt buộc để tính perplexity
        )
        print("  ✅ GGUF Model loaded successfully!")
        return llm
    except Exception as e:
        print(f"  ❌ Failed to load GGUF model: {e}")
        return None