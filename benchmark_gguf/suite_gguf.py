import psutil
import time
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from llama_cpp import Llama
import os
import subprocess

from . import tasks_gguf as tasks

class NpEncoder(json.JSONEncoder):
    """
    JSON encoder tùy chỉnh để xử lý các kiểu dữ liệu của NumPy.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def _get_nvidia_vram_usage() -> float:
    """
    Lấy mức sử dụng VRAM của tiến trình hiện tại bằng nvidia-smi.
    Chỉ hoạt động trên GPU NVIDIA.

    Returns:
        Mức sử dụng VRAM tính bằng GB. Trả về 0.0 nếu không thể lấy được.
    """
    try:
        # Lấy PID của tiến trình hiện tại
        pid = os.getpid()
        # Tạo câu lệnh nvidia-smi để truy vấn bộ nhớ GPU được sử dụng bởi PID cụ thể
        cmd = [
            'nvidia-smi',
            '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Phân tích output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            parts = line.split(', ')
            if len(parts) == 2 and int(parts[0]) == pid:
                # Giá trị trả về là MiB, chuyển đổi sang GB
                vram_mib = int(parts[1])
                return vram_mib / 1024
        return 0.0 # Trả về 0 nếu không tìm thấy PID trong output
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        return 0.0


class EnhancedBenchmarkSuiteGGUF:
    def __init__(self, llm: Llama, model_id: str, run_name: str, model_hash: str):
        self.llm = llm
        self.model_id = model_id
        self.run_name = run_name
        self.model_hash = model_hash
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        # Kiểm tra một lần xem có thể lấy thông tin VRAM không
        self.vram_gb_at_start = _get_nvidia_vram_usage()
        if self.vram_gb_at_start > 0:
            print(f"  ✅ Detected NVIDIA GPU. Initial VRAM usage: {self.vram_gb_at_start:.2f} GB")
        else:
            print("  ⚠️ Could not detect NVIDIA VRAM usage via nvidia-smi. VRAM will be reported as 0.")


    def _get_memory_usage(self) -> Dict[str, float]:
        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1024**3
        # Sử dụng giá trị đã kiểm tra lúc khởi tạo
        gpu_allocated_gb = self.vram_gb_at_start 
        return {'ram_gb': ram_gb, 'gpu_allocated_gb': gpu_allocated_gb}

    def _generate_response(self, prompt: str, task_name: str = "") -> Dict:
        print(f"    🔄 Generating response for: {task_name}")
        start_time = time.time()
        completion = self.llm.create_completion(
            prompt, max_tokens=4096, temperature=0.6, top_p=0.9,
            stop=self.stop_tokens, echo=False
        )
        end_time = time.time()
        
        response_text = completion['choices'][0]['text'].strip()
        usage_info = completion['usage']
        
        generation_time = end_time - start_time
        tokens_generated = usage_info['completion_tokens']
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Lấy thông tin VRAM (sẽ là giá trị tĩnh được đo khi tải model)
        vram_usage = self._get_memory_usage()['gpu_allocated_gb']

        return {
            'prompt': prompt,
            'response': response_text,
            'generation_time': round(generation_time, 2),
            'tokens_generated': tokens_generated,
            'tokens_per_second': round(tokens_per_second, 2),
            'vram_usage_gb': round(vram_usage, 2) # Trả về giá trị VRAM đã đo
        }

    def _run_single_generation_task(self, task_name: str, prompts: List[str]) -> Dict:
        print(f"  📊 Benchmarking Generation Task: {task_name}")
        results = [self._generate_response(p, f"{task_name} #{i+1}") for i, p in enumerate(prompts)]
        avg_tps = np.mean([r['tokens_per_second'] for r in results])
        self.llm.reset()
        return {'task': task_name, 'avg_tokens_per_second': avg_tps, 'tests': results} # Giữ np.float để encoder xử lý

    def run_full_benchmark(self, mmlu_dir: Path, PERPLEXITY_CATEGORIES: list, traditional_tasks: dict):
        print("\n🚀 Starting GGUF Model Comprehensive Benchmark")
        print("=" * 80)

        mmlu_results = tasks.run_mmlu_test(self.llm, mmlu_dir)
        perplexity_results = tasks.run_perplexity_test(self.llm, PERPLEXITY_CATEGORIES)
        
        generation_results = []
        for name, prompts in traditional_tasks.items():
            generation_results.append(self._run_single_generation_task(name, prompts))

        overall_stats = {
            'avg_token_speed': np.mean([r['avg_tokens_per_second'] for r in generation_results]) if generation_results else 0,
            'mmlu_accuracy': mmlu_results.get('overall_accuracy', 0),
            'avg_perplexity': perplexity_results.get('overall_average_perplexity', 0),
            'vram_usage_gb': self.vram_gb_at_start # Thêm thông số này vào kết quả tổng thể
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
        
        print("\n" + "="*80 + "\n📊 GGUF MODEL ENHANCED BENCHMARK RESULTS SUMMARY\n" + "="*80)
        print(f"📦 Model ID: {self.model_id}")
        print(f"⚙️ Run Type: {self.run_name}")
        print(f"🎯 Model Hash: {results['model_hash']}")
        print("-" * 80)
        # Thêm thông tin VRAM vào bản tóm tắt
        if stats.get('vram_usage_gb', 0) > 0:
            print(f"💾 VRAM Usage: {float(stats['vram_usage_gb']):.2f} GB")
        print(f"📈 Average Token Speed (Generation): {float(stats['avg_token_speed']):.2f} tokens/sec")
        print(f"🧠 MMLU Accuracy: {float(stats['mmlu_accuracy']):.2f}%")
        print(f"📊 Overall Average Perplexity: {float(stats['avg_perplexity']):.4f}")
        
        if 'category_results' in results['perplexity_results']:
            print("\n   --- Perplexity by Category ---")
            for category, data in results['perplexity_results']['category_results'].items():
                print(f"   - {category:<25}: {float(data['average_perplexity']):.4f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = self.model_id.replace('/', '_')
        filename = f"benchmark_gguf_{safe_model_name}_{self.run_name}_{timestamp}.json"
        filepath = Path("benchmark_results") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            results['run_name'] = self.run_name
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            
        print(f"\n💾 Full detailed results saved to: {filepath}")