import warnings
import traceback
import os

from llama_cpp import Llama 
from benchmark_gguf.suite_gguf import EnhancedBenchmarkSuiteGGUF
from config.test_cases import MMLU_TEST_DIR, PERPLEXITY_CATEGORIES, TRADITIONAL_GENERATION_TASKS
from benchmark_gguf.utils_gguf import get_model_hash

warnings.filterwarnings("ignore")

LOCAL_MODEL_PATH = "./models/Qwen3-8B-gguf-ewq-tensorwise.gguf" 
GGUF_CONFIG = {
    "logits_all":True,
    "n_gpu_layers": -1,  # Offload tất cả các layer lên GPU. Đặt là 0 nếu chỉ dùng CPU.
    "n_ctx": 8192,       # Kích thước ngữ cảnh
    "verbose": False     # Tắt log chi tiết của llama.cpp
}


def main():
    # Lấy tên file từ đường dẫn để đặt tên cho lần chạy benchmark
    model_filename = os.path.basename(LOCAL_MODEL_PATH)
    run_name = model_filename.replace(".gguf", "")

    print(f"🔍 Starting Benchmark for LOCAL GGUF Model: {LOCAL_MODEL_PATH}")
    print(f"🚀 Run Name: {run_name}")
    print("=" * 80)
    
    try:
        # 1. Xác định model và định danh
        # Đối với model offline, đường dẫn file chính là định danh duy nhất
        model_identifier = LOCAL_MODEL_PATH
        print(f"🔑 Model Identifier: {model_identifier}")
        
        # 2. Tải model GGUF trực tiếp từ file
        # Đây là thay đổi quan trọng nhất!
        print(f"⏳ Loading GGUF model from local path: {LOCAL_MODEL_PATH}")
        llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            **GGUF_CONFIG
        )
        
        print("✅ GGUF model is loaded and ready for benchmarking!")
        
        # 3. Khởi tạo và chạy bộ benchmark
        benchmark_suite = EnhancedBenchmarkSuiteGGUF(
            llm=llm,
            model_id=model_identifier,
            run_name=run_name,
            model_hash=get_model_hash(run_name, GGUF_CONFIG)  # Sử dụng run_name và GGUF_CONFIG để tạo hash duy nhất
        )
        
        # Truyền các test case từ file config vào
        benchmark_suite.run_full_benchmark(
            mmlu_dir=MMLU_TEST_DIR,
            PERPLEXITY_CATEGORIES=PERPLEXITY_CATEGORIES,
            traditional_tasks=TRADITIONAL_GENERATION_TASKS
        )
        
        print(f"\n🎊 Benchmark for {model_filename} completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()