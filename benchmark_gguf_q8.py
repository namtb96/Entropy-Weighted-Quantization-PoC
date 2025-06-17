import warnings
import traceback

from benchmark_gguf.loader_gguf import load_gguf_model
from benchmark_gguf.suite_gguf import EnhancedBenchmarkSuiteGGUF
from benchmark_gguf.utils_gguf import get_model_hash
from config.test_cases import MMLU_TEST_DIR, PERPLEXITY_CATEGORIES, TRADITIONAL_GENERATION_TASKS

warnings.filterwarnings("ignore")

# === CẤU HÌNH CHO MODEL GGUF ===
# Thay đổi repo ID và tên file cho phù hợp với model bạn muốn test
MODEL_REPO_ID = "unsloth/Llama-3.1-8B-Instruct-GGUF"
MODEL_FILE = "unsloth/Llama-3.1-8B-Instruct-GGUF" 
MODEL_CACHE_DIR = "./models"

# Cấu hình cho Llama.cpp
GGUF_CONFIG = {
    "n_gpu_layers": -1,  # Offload tất cả các layer lên GPU. Đặt là 0 nếu chỉ dùng CPU.
    "n_ctx": 4096,       # Kích thước ngữ cảnh
    "verbose": False     # Tắt log chi tiết của llama.cpp
}

def main():
    run_name = MODEL_FILE.replace(".gguf", "")
    print(f"🔍 Starting Benchmark for GGUF Model: {MODEL_REPO_ID}/{MODEL_FILE}")
    print(f"🚀 Run Name: {run_name}")
    print("=" * 80)
    
    try:
        # 1. Xác định model và hash
        model_config = {'repo_id': MODEL_REPO_ID, 'file': MODEL_FILE, 'config': GGUF_CONFIG}
        model_hash = get_model_hash(MODEL_REPO_ID, model_config)
        print(f"🔑 Model Config Hash: {model_hash}")
        
        # 2. Tải model GGUF
        llm = load_gguf_model(
            repo_id=MODEL_REPO_ID,
            model_file=MODEL_FILE,
            model_cache_dir=MODEL_CACHE_DIR,
            **GGUF_CONFIG
        )
        
        if llm is None:
            print("❌ Failed to load GGUF model!"); return
        
        print("✅ GGUF model is loaded and ready for benchmarking!")
        
        # 3. Khởi tạo và chạy bộ benchmark
        benchmark_suite = EnhancedBenchmarkSuiteGGUF(
            llm=llm,
            model_id=f"{MODEL_REPO_ID}/{MODEL_FILE}",
            run_name=run_name,
            model_hash=model_hash
        )
        
        # Truyền các test case từ file config vào
        benchmark_suite.run_full_benchmark(
            mmlu_dir=MMLU_TEST_DIR,
            PERPLEXITY_CATEGORIES=PERPLEXITY_CATEGORIES,
            traditional_tasks=TRADITIONAL_GENERATION_TASKS
        )
        
        print(f"\n🎊 Benchmark for {MODEL_FILE} completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()