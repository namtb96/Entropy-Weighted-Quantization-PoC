import warnings
import traceback

from benchmark.quantization import load_quantized_model, check_quantized_model_exists
from benchmark.suite import EnhancedBenchmarkSuite
from benchmark.utils import get_model_hash
from config.test_cases import MMLU_TEST_DIR, PERPLEXITY_CATEGORIES, TRADITIONAL_GENERATION_TASKS

warnings.filterwarnings("ignore")

# === CẤU HÌNH CHO MODEL CỤ THỂ NÀY ===
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_CACHE_DIR = "./models"
ENTROPY_THRESHOLD_FACTOR = 0.65 # Ngưỡng entropy để quyết định lượng tử hóa

def main():
    print(f"🔍 Starting Benchmark for Model: {MODEL_ID}")
    print("=" * 80)
    
    try:
        # 1. Xác định model và hash
        model_config = {'base_model': MODEL_ID, 'entropy_factor': ENTROPY_THRESHOLD_FACTOR, 'quant_method': 'ewq-bitsandbytes'}
        model_hash = get_model_hash(MODEL_ID, model_config)
        run_name = model_config['quant_method']
        print(f"🔑 Model Config Hash: {model_hash}")
        
        if not check_quantized_model_exists(model_hash):
            print(f"❌ Error: Quantization plan not found for hash '{model_hash}'.")
            print("💡 Please run the 'create_quantization_plan.py' script first.")
            return
        
        # 2. Tải model đã lượng tử hóa
        print("✅ Found quantization plan. Proceeding to load and quantize model.")
        model, tokenizer = load_quantized_model(MODEL_ID, model_hash, MODEL_CACHE_DIR)
        
        if model is None or tokenizer is None:
            print("❌ Failed to load and quantize model!"); return
        
        print("✅ Quantized model is on GPU and ready for benchmarking!")
        
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
        
        print(f"\n🎊 Benchmark for {MODEL_ID} completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()