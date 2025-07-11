import warnings
import traceback
import os

from benchmark.quantization import load_original_model
from benchmark.suite import EnhancedBenchmarkSuite
from benchmark.utils import get_model_hash
from config.test_cases import MMLU_TEST_DIR, PERPLEXITY_CATEGORIES, TRADITIONAL_GENERATION_TASKS

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# === CẤU HÌNH CHO MODEL GỐC ===
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_CACHE_DIR = "./models"

def main():
    print(f"🔍 Starting Benchmark for ORIGINAL (FP16) Model: {MODEL_ID}")
    print("=" * 80)
    
    try:
        # 1. Xác định model và hash
        # Tạo một config riêng cho model gốc để hash và tên file kết quả không bị trùng
        model_config = {'base_model': MODEL_ID, 'quant_method': 'original-fp16'}
        model_hash = get_model_hash(MODEL_ID, model_config)
        run_name = model_config['quant_method']
        print(f"🔑 Model Config Hash (Original): {model_hash}")
        
        # 2. Tải model gốc (không cần kiểm tra plan)
        print("✅ Proceeding to load original model.")
        model, tokenizer = load_original_model(MODEL_ID, MODEL_CACHE_DIR)
        
        if model is None or tokenizer is None:
            print("❌ Failed to load original model!"); return
        
        print("✅ Original model is on GPU (or CPU if no CUDA) and ready for benchmarking!")
        
        # 3. Khởi tạo và chạy bộ benchmark
        benchmark_suite = EnhancedBenchmarkSuite(
            model=model, 
            tokenizer=tokenizer, 
            model_id=MODEL_ID, 
            run_name=run_name, 
            model_hash=model_hash
        )
        
        # Truyền các test case từ file config vào
        benchmark_suite.run_full_benchmark(
            mmlu_dir=MMLU_TEST_DIR,
            PERPLEXITY_CATEGORIES=PERPLEXITY_CATEGORIES,
            traditional_tasks=TRADITIONAL_GENERATION_TASKS
        )
        
        print(f"\n🎊 Benchmark for ORIGINAL model {MODEL_ID} completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()