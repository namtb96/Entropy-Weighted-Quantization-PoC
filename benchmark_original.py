#!/usr/bin/env python3
"""
Original Model Benchmark System (Baseline)

Hệ thống này được thiết kế để đánh giá hiệu năng của MODEL GỐC (chưa qua lượng tử hóa)
để tạo một đường cơ sở (baseline) so sánh.

Quy trình:
1. Tải model gốc trực tiếp từ Hugging Face lên GPU bằng phương pháp tiêu chuẩn.
2. Chạy bộ benchmark toàn diện với đầy đủ logging.
"""
import torch
import torch.nn as nn
import time
import psutil
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from datetime import datetime
import gc
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# === Cấu hình (Phải khớp với model gốc bạn muốn so sánh) ===
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_CACHE_DIR = "./models"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# === HÀM TẢI MODEL GỐC ===

def load_original_model(model_id: str) -> Tuple[Optional[nn.Module], Optional[AutoTokenizer]]:
    """Tải model gốc (unquantized) trực tiếp lên GPU."""
    
    print("  📥 Loading original base model to GPU with `device_map='auto'`...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",  # Tải trực tiếp lên GPU
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    except Exception as e:
        print(f"❌ Failed to load original model: {e}")
        return None, None


# === BENCHMARK SUITE (GIỮ NGUYÊN HOÀN TOÀN) ===

class BenchmarkSuite:
    def __init__(self, model, tokenizer, model_identifier: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_identifier = model_identifier
        self.device = next(model.parameters()).device
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Lấy thông tin sử dụng bộ nhớ (RAM & VRAM)."""
        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1024**3
        gpu_allocated_gb = 0
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
        return {'ram_gb': ram_gb, 'gpu_allocated_gb': gpu_allocated_gb}
    
    def generate_response(self, prompt: str, task_name: str = "") -> Dict:
        """Sinh response và đo lường hiệu năng chi tiết."""
        print(f"    🔄 Generating response for: {task_name}")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        torch.cuda.synchronize()
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
              **inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9,
              repetition_penalty=1.1, pad_token_id=self.tokenizer.pad_token_id,
              eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        generation_time = end_time - start_time
        tokens_generated = len(response_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        return {
            'prompt': prompt, 'response': response, 'generation_time': round(generation_time, 2),
            'tokens_generated': tokens_generated, 'tokens_per_second': round(tokens_per_second, 2),
            'vram_usage_gb': round(memory_after['gpu_allocated_gb'], 2)
        }
        
    def _run_single_benchmark(self, task_name: str, prompts: List[str]) -> Dict:
        print(f"  📊 Benchmarking: {task_name}")
        results = [self.generate_response(p, f"{task_name} #{i+1}") for i, p in enumerate(prompts)]
        time.sleep(1)
        avg_time = np.mean([r['generation_time'] for r in results])
        avg_tps = np.mean([r['tokens_per_second'] for r in results])
        return {'task': task_name, 'avg_generation_time': round(avg_time, 2),
                'avg_tokens_per_second': round(avg_tps, 2), 'tests': results}

    def run_full_benchmark(self) -> Dict:
        """Chạy toàn bộ bộ benchmark."""
        print("\n🚀 Starting Original Model Comprehensive Benchmark")
        print("=" * 60)
        
        tasks = {
            "Code Generation": [
                "Viết một script Python sử dụng thư viện Pandas để đọc file CSV có tên `sales_data.csv` với các cột 'Date', 'Product', 'Revenue'. Script cần tính tổng doanh thu theo từng sản phẩm và xuất kết quả ra một file CSV mới có tên `revenue_by_product.csv`.",
                "Tạo một component React functional bằng TypeScript tên là `UserProfile`. Component này nhận vào props là `name` (string), `age` (number), và `avatarUrl` (string), sau đó hiển thị thông tin này một cách có cấu trúc."
            ],
            "Math Problem Solving": [
                "Một bể nước có hai vòi. Vòi thứ nhất chảy một mình thì đầy bể trong 4 giờ. Vòi thứ hai chảy một mình thì đầy bể trong 6 giờ. Nếu mở cả hai vòi cùng một lúc khi bể cạn, hỏi sau bao lâu thì bể sẽ đầy? Trình bày các bước giải chi tiết.",
                "Một người gửi tiết kiệm 500 triệu đồng với lãi suất kép 6.5% mỗi năm. Hỏi sau 5 năm, người đó sẽ nhận được cả vốn lẫn lãi là bao nhiêu tiền? Yêu cầu trình bày công thức và các bước tính toán."
            ],
            "Text Summarization": [
                "Hãy tóm tắt đoạn văn sau thành 3 ý chính: 'Các công nghệ thu giữ carbon (Carbon Capture Technologies) đang nổi lên như một giải pháp tiềm năng trong cuộc chiến chống biến đổi khí hậu. Các phương pháp chính bao gồm thu giữ sau đốt cháy, thu giữ trước đốt cháy và thu giữ từ không khí trực tiếp (DAC). Mặc dù có tiềm năng lớn, công nghệ này vẫn đối mặt với những thách thức về chi phí vận hành cao, hiệu quả năng lượng và vấn đề lưu trữ carbon an toàn trong dài hạn. Các chính phủ và tập đoàn lớn đang đầu tư hàng tỷ USD vào R&D để cải thiện hiệu quả và giảm giá thành, hy vọng biến nó thành một công cụ chủ chốt vào năm 2050.'",
                "Tóm tắt ngắn gọn những sự kiện chính và ý nghĩa lịch sử của cuộc Cách mạng Công nghiệp lần thứ nhất, tập trung vào các phát minh quan trọng và tác động của nó đến xã hội."
            ],
            "Creative Writing": [
                "Viết đoạn mở đầu cho một câu chuyện ngắn thuộc thể loại khoa học viễn tưởng, trong đó nhân vật chính là một nhà thực vật học sống trên Sao Hỏa, người vừa phát hiện ra một loài cây có khả năng giao tiếp bằng ánh sáng.",
                "Viết một bài văn ngắn (khoảng 150 từ) miêu tả cảnh một phiên chợ nổi trên sông Cửu Long vào buổi sáng sớm, tập trung vào âm thanh, màu sắc và mùi vị đặc trưng."
            ],
            "Question Answering": [
                "Giải thích sự khác biệt cơ bản giữa năng lượng hạt nhân phân hạch (nuclear fission) và tổng hợp hạt nhân (nuclear fusion). Loại nào hiện đang được sử dụng trong các nhà máy điện?",
                "Con đường tơ lụa là gì và nó có vai trò quan trọng như thế nào đối với sự phát triển của các nền văn minh cổ đại?"
            ],
            "Language Translation & Nuance": [
                "Dịch đoạn văn sau sang tiếng Việt, chú ý giữ văn phong chuyên nghiệp: 'Our team is conducting a comprehensive due diligence process to assess the viability of the potential acquisition. Key performance indicators (KPIs) and financial statements are under rigorous scrutiny.'",
                "Giải thích ý nghĩa và tìm câu thành ngữ tiếng Anh có nghĩa tương đương với câu 'Nước đến chân mới nhảy'."
            ],
            "Reasoning & Logic": [
                "Có ba chiếc hộp. Một hộp chứa toàn bi đỏ, một hộp chứa toàn bi xanh, và một hộp chứa lẫn lộn cả bi đỏ và bi xanh. Cả ba hộp đều bị dán nhãn sai. Bạn chỉ được phép lấy ra một viên bi từ một hộp duy nhất (không được nhìn vào bên trong). Làm thế nào để xác định chính xác nội dung của cả ba hộp? Hãy giải thích quá trình suy luận của bạn.",
                "Một nghiên cứu cho thấy những thành phố có nhiều cửa hàng kem nhất cũng có tỷ lệ tội phạm cao nhất. Có phải ăn kem gây ra tội phạm không? Hãy giải thích về mối tương quan và quan hệ nhân quả (correlation vs. causation) trong trường hợp này."
            ],
            "Technical Explanation": [
                "Hãy giải thích cho một người không rành về công nghệ về khái niệm 'Điện toán đám mây' (Cloud Computing) là gì. Sử dụng ví dụ về Google Drive hoặc iCloud để minh họa.",
                "Giải thích một cách đơn giản quá trình quang hợp ở thực vật diễn ra như thế nào và tại sao nó lại quan trọng đối với sự sống trên Trái Đất."
            ]
        }

        benchmark_results = []
        for name, prompts in tasks.items():
            benchmark_results.append(self._run_single_benchmark(name, prompts))
            gc.collect(); torch.cuda.empty_cache()

        overall_tps = [r['avg_tokens_per_second'] for r in benchmark_results]
        overall_stats = {'avg_token_speed': round(np.mean(overall_tps), 2) if overall_tps else 0}
        
        return {'model_identifier': self.model_identifier, 'timestamp': datetime.now().isoformat(),
                'overall_stats': overall_stats, 'benchmark_results': benchmark_results}

    def save_and_print_summary(self, results: Dict):
        """Lưu và in kết quả benchmark."""
        stats = results['overall_stats']
        print("\n" + "="*60 + "\n📊 ORIGINAL MODEL BENCHMARK RESULTS SUMMARY\n" + "="*60)
        print(f"🎯 Model: {results['model_identifier']}")
        print(f"📈 Average Token Speed (Overall): {stats['avg_token_speed']:.2f} tokens/sec")
        
        print("\n📋 Task Performance:")
        for result in results['benchmark_results']:
            vram = result['tests'][0]['vram_usage_gb'] if result['tests'] else 'N/A'
            print(f"  - {result['task']:<25}: {result['avg_tokens_per_second']:.2f} tok/s @ {vram} GB VRAM")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"original_benchmark_{self.model_identifier.replace('/', '_')}_{timestamp}.json"
        filepath = Path("benchmark_results") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full detailed results saved to: {filepath}")

# === HÀM MAIN ĐỂ CHẠY BENCHMARK ===
def main():
    print("🔍 Original Model Benchmark System (Baseline)")
    print("=" * 60)
    
    try:
        model, tokenizer = load_original_model(MODEL_ID)
        
        if model is None or tokenizer is None: return
        
        print("✅ Original model is on GPU and ready for benchmarking!")
        
        model_identifier = MODEL_ID + "-original"
        benchmark_suite = BenchmarkSuite(model, tokenizer, model_identifier)
        results = benchmark_suite.run_full_benchmark()
        benchmark_suite.save_and_print_summary(results)
        
        print(f"\n🎊 Benchmark for original model completed successfully!")
        
    except Exception as e:
        print(f"❌ An unexpected error occurred during benchmark: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()