# ewq_benchmark/suite.py

import torch
import psutil
import time
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import gc
from typing import Dict, List

from . import tasks

class EnhancedBenchmarkSuite:
    def __init__(self, model, tokenizer, model_id: str, run_name: str, model_hash: str):
        """
        Khởi tạo bộ benchmark.

        Args:
            model: Model ngôn ngữ.
            tokenizer: Tokenizer tương ứng.
            model_id (str): ID của model (ví dụ: 'Qwen/Qwen3-8B').
            run_name (str): Tên mô tả cho lần chạy này (ví dụ: 'ewq-bitsandbytes' hoặc 'original-fp16').
            model_hash (str): Hash duy nhất của cấu hình model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.run_name = run_name
        self.model_hash = model_hash
        self.device = next(model.parameters()).device

    def _get_memory_usage(self) -> Dict[str, float]:
        process = psutil.Process(); ram_gb = process.memory_info().rss / 1024**3
        gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        return {'ram_gb': ram_gb, 'gpu_allocated_gb': gpu_allocated_gb}

    def _generate_response(self, prompt: str, task_name: str = "") -> Dict:
      print(f"    🔄 Generating response for: {task_name}")
      
      inputs = self.tokenizer(
          prompt, 
          return_tensors="pt", 
          truncation=True, 
          max_length=4096
      ).to(self.device)
      
      start_time = time.time()
      
      with torch.no_grad():
          outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096, 
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
          )
      
      end_time = time.time()
      
      input_length = inputs['input_ids'].shape[1]
      response_tokens = outputs[0][input_length:]
      
      # <<< THÊM BƯỚC XỬ LÝ HẬU KỲ Ở ĐÂY >>>
      # Decode toàn bộ chuỗi để tìm token kết thúc dưới dạng text
      full_response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
      
      # Lấy token kết thúc từ tokenizer
      eos_token_text = self.tokenizer.decode(self.tokenizer.eos_token_id)

      # Cắt chuỗi tại vị trí của token kết thúc đầu tiên
      if eos_token_text in full_response_text:
          clean_response_text = full_response_text.split(eos_token_text)[0]
      else:
          clean_response_text = full_response_text

      # Decode lại lần nữa với skip_special_tokens=True để dọn dẹp
      final_response = self.tokenizer.decode(self.tokenizer.encode(clean_response_text), skip_special_tokens=True).strip()

      generation_time = end_time - start_time
      tokens_generated = len(response_tokens) # Vẫn đo lường tổng số token được sinh ra
      tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
      
      return {
          'prompt': prompt,
          'response': final_response, # Trả về response đã được làm sạch
          'full_raw_response': full_response_text, # Có thể giữ lại để debug
          'generation_time': round(generation_time, 2),
          'tokens_generated': tokens_generated,
          'tokens_per_second': round(tokens_per_second, 2),
          'vram_usage_gb': round(self._get_memory_usage()['gpu_allocated_gb'], 2)
      }


    def _run_single_generation_task(self, task_name: str, prompts: List[str]) -> Dict:
        print(f"  📊 Benchmarking Generation Task: {task_name}")
        results = [self._generate_response(p, f"{task_name} #{i+1}") for i, p in enumerate(prompts)]
        avg_tps = np.mean([r['tokens_per_second'] for r in results])
        return {'task': task_name, 'avg_tokens_per_second': round(avg_tps, 2), 'tests': results}

    def run_full_benchmark(self, mmlu_dir: Path, PERPLEXITY_CATEGORIES: list, traditional_tasks: dict):
        """Chạy toàn bộ bộ benchmark, nhận các test case làm tham số."""
        print("\n🚀 Starting EWQ Model Comprehensive Benchmark")
        print("=" * 80)

        # Gọi các hàm task từ module tasks.py
        mmlu_results = tasks.run_mmlu_test(self.model, self.tokenizer, self.device, mmlu_dir)
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        perplexity_results = tasks.run_perplexity_test(self.model, self.tokenizer, self.device, PERPLEXITY_CATEGORIES)
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        generation_results = []
        for name, prompts in traditional_tasks.items():
            generation_results.append(self._run_single_generation_task(name, prompts))
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        overall_stats = {
            'avg_token_speed': round(np.mean([r['avg_tokens_per_second'] for r in generation_results]), 2) if generation_results else 0,
            'mmlu_accuracy': mmlu_results.get('overall_accuracy', 0),
            'avg_perplexity': perplexity_results.get('overall_average_perplexity', 0)
        }
        
        full_results = {
            'model_hash': self.model_hash,
            'timestamp': datetime.now().isoformat(),
            'overall_stats': overall_stats,
            'mmlu_results': mmlu_results,
            'perplexity_results': perplexity_results,
            'generation_results': generation_results
        }
        self.save_and_print_summary(full_results)
        return full_results
    
    def save_and_print_summary(self, results: Dict):
        """Lưu và in kết quả benchmark với tên file dễ đọc."""
        stats = results['overall_stats']
        
        # In ra bản tóm tắt kết quả
        print("\n" + "="*80 + "\n📊 EWQ MODEL ENHANCED BENCHMARK RESULTS SUMMARY\n" + "="*80)
        print(f"📦 Model ID: {self.model_id}")
        print(f"⚙️ Run Type: {self.run_name}") # Thêm thông tin loại run
        print(f"🎯 Model Hash: {results['model_hash']}") # Vẫn giữ hash trong log để định danh
        print("-" * 80)
        print(f"📈 Average Token Speed (Generation): {stats['avg_token_speed']:.2f} tokens/sec")
        print(f"🧠 MMLU Accuracy: {stats['mmlu_accuracy']:.2f}%")
        print(f"📊 Overall Average Perplexity: {stats['avg_perplexity']:.4f}")
        
        # In chi tiết kết quả perplexity theo danh mục
        if 'category_results' in results['perplexity_results']:
            print("\n   --- Perplexity by Category ---")
            for category, data in results['perplexity_results']['category_results'].items():
                print(f"   - {category:<25}: {data['average_perplexity']:.4f}")
        
        # Tạo tên file dễ đọc
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Làm sạch tên model để dùng trong tên file (loại bỏ dấu gạch chéo)
        safe_model_name = self.model_id.split('/')[-1]
        
        # Tạo tên file mới rõ ràng hơn
        filename = f"benchmark_{safe_model_name}_{self.run_name}_{timestamp}.json"
        filepath = Path("benchmark_results") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Thêm run_name vào nội dung file JSON để tham khảo
            results['run_name'] = self.run_name
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 Full detailed results saved to: {filepath}")