#!/usr/bin/env python3
"""
Original Model Benchmark System - WITHOUT EWQ Quantization

Script này dùng để benchmark model gốc (không lượng tử hóa) để so sánh với kết quả EWQ.
Bao gồm tất cả các bài test: MMLU, Perplexity và Generation tasks.

Quy trình:
1. Tải model gốc lên GPU với precision cao nhất có thể.
2. Chạy toàn bộ bộ benchmark với đầy đủ logging.
3. Lưu kết quả để so sánh với EWQ benchmark.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import gc
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === Cấu hình ===
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_CACHE_DIR = "./models"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# === MMLU Test Questions (Sample subset for efficiency) ===
MMLU_QUESTIONS = {
    "abstract_algebra": [
        {
            "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
            "choices": ["0", "4", "2", "6"],
            "answer": 1
        },
        {
            "question": "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the index of <p> in S_5.",
            "choices": ["8", "2", "24", "120"],
            "answer": 2
        }
    ],
    "anatomy": [
        {
            "question": "Which of the following is the body cavity that contains the pituitary gland?",
            "choices": ["Abdominal", "Cranial", "Pleural", "Spinal"],
            "answer": 1
        },
        {
            "question": "What is the embryological origin of the hyoid bone?",
            "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
            "answer": 3
        }
    ],
    "astronomy": [
        {
            "question": "How long does it take for light to travel from the Sun to the Earth?",
            "choices": ["8 minutes", "1 hour", "1 day", "1 year"],
            "answer": 0
        },
        {
            "question": "Where do most short-period comets come from and how do we know?",
            "choices": ["The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.", "The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.", "The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.", "The Oort cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort cloud."],
            "answer": 0
        }
    ],
    "business_ethics": [
        {
            "question": "According to Kant's categorical imperative, we should:",
            "choices": ["Act only according to that maxim whereby you can at the same time will that it should become a universal law.", "Act to maximize happiness for the greatest number of people.", "Act only according to that maxim which you can at the same time will to be a universal law of nature.", "Act to benefit yourself, as long as you don't harm others."],
            "answer": 0
        },
        {
            "question": "What is the difference between a stakeholder and a shareholder?",
            "choices": ["Stakeholders own shares in the company, while shareholders have an interest in the company.", "Stakeholders have an interest in the company, while shareholders own shares in the company.", "Stakeholders and shareholders are the same thing.", "Stakeholders are employees, while shareholders are customers."],
            "answer": 1
        }
    ],
    "clinical_knowledge": [
        {
            "question": "Glycolysis is the name given to the pathway involving the conversion of:",
            "choices": ["glycogen to glucose-1-phosphate.", "glycogen or starch to fructose.", "glycogen or starch to glucose or glucose-1-phosphate.", "glucose to pyruvate or lactate."],
            "answer": 3
        },
        {
            "question": "A patient has been on the operating table for four hours. How long may it take for any pressure damage to be reversible?",
            "choices": ["30 minutes.", "2 hours.", "12 hours.", "24 hours."],
            "answer": 2
        }
    ]
}

# === Perplexity Test Passages ===
PERPLEXITY_PASSAGES = [
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once.",
    "In the beginning was the Word, and the Word was with God, and the Word was God. He was in the beginning with God.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data.",
    "The mitochondria is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
    "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities since the 1800s.",
    "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles.",
    "The Internet has revolutionized communication, commerce, and access to information, connecting billions of people worldwide.",
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
]

def get_model_hash(model_id: str) -> str:
    """Tạo hash cho model gốc (không quantization)."""
    config_str = f"{model_id}-original-no-quantization"
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def load_original_model(model_id: str) -> Tuple[Optional[nn.Module], Optional[AutoTokenizer]]:
    """Tải model gốc với precision cao nhất có thể."""
    print("  📥 Loading original model to GPU...")
    
    # Kiểm tra VRAM available
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  💾 Available VRAM: {total_vram:.2f} GB")
        
        # Chọn precision dựa trên VRAM
        if total_vram >= 16:
            torch_dtype = torch.float32
            print("  🎯 Using float32 precision (highest quality)")
        elif total_vram >= 12:
            torch_dtype = torch.float16
            print("  🎯 Using float16 precision (balanced)")
        else:
            torch_dtype = torch.float16
            print("  ⚠️ Using float16 precision (limited VRAM)")
            
        device_map = "auto"
    else:
        print("  ⚠️ WARNING: No CUDA device found. Model will run on CPU.")
        torch_dtype = torch.float32
        device_map = "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True if torch.cuda.is_available() else False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("  ✅ Original model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"  ❌ Failed to load original model: {e}")
        return None, None


# === ORIGINAL MODEL BENCHMARK SUITE ===

class OriginalModelBenchmarkSuite:
    def __init__(self, model, tokenizer, model_hash: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_hash = model_hash
        self.device = next(model.parameters()).device
        self.model_precision = str(next(model.parameters()).dtype)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Lấy thông tin sử dụng bộ nhớ (RAM & VRAM)."""
        process = psutil.Process()
        ram_gb = process.memory_info().rss / 1024**3
        
        gpu_allocated_gb = 0
        gpu_reserved_gb = 0
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
        
        return {
            'ram_gb': ram_gb, 
            'gpu_allocated_gb': gpu_allocated_gb,
            'gpu_reserved_gb': gpu_reserved_gb
        }
    
    def generate_response(self, prompt: str, task_name: str = "") -> Dict:
        """Sinh response và đo lường hiệu năng chi tiết."""
        print(f"    🔄 Generating response for: {task_name}")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
              **inputs,
              max_new_tokens=512,
              do_sample=True,
              temperature=0.6,
              top_p=0.9,
              repetition_penalty=1.1,
              pad_token_id=self.tokenizer.pad_token_id,
              eos_token_id=self.tokenizer.eos_token_id,
              use_cache=True 
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        generation_time = end_time - start_time
        tokens_generated = len(response_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        return {
            'prompt': prompt,
            'response': response,
            'generation_time': round(generation_time, 2),
            'tokens_generated': tokens_generated,
            'tokens_per_second': round(tokens_per_second, 2),
            'vram_allocated_gb': round(memory_after['gpu_allocated_gb'], 2),
            'vram_reserved_gb': round(memory_after['gpu_reserved_gb'], 2)
        }

    def run_mmlu_test(self) -> Dict:
        """Chạy MMLU (Massive Multitask Language Understanding) test."""
        print("  🧠 Running MMLU Test...")
        
        correct_answers = 0
        total_questions = 0
        subject_results = {}
        
        for subject, questions in MMLU_QUESTIONS.items():
            print(f"    📚 Testing subject: {subject}")
            subject_correct = 0
            
            for i, q in enumerate(questions):
                # Format câu hỏi theo chuẩn MMLU
                choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(q['choices'])])
                prompt = f"Question: {q['question']}\n{choices_text}\nAnswer:"
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=5,  # Chỉ cần 1 ký tự cho đáp án
                        do_sample=False,   # Greedy decoding cho tính nhất quán
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                
                # Kiểm tra đáp án
                predicted_letter = response[0].upper() if response and response[0].upper() in 'ABCD' else 'X'
                correct_letter = chr(65 + q['answer'])
                
                if predicted_letter == correct_letter:
                    subject_correct += 1
                    correct_answers += 1
                
                total_questions += 1
                
                print(f"      Q{i+1}: Predicted={predicted_letter}, Correct={correct_letter}, {'✓' if predicted_letter == correct_letter else '✗'}")
            
            subject_accuracy = subject_correct / len(questions) * 100
            subject_results[subject] = {
                'correct': subject_correct,
                'total': len(questions),
                'accuracy': round(subject_accuracy, 2)
            }
            
            print(f"    📊 {subject} accuracy: {subject_accuracy:.2f}%")
        
        overall_accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0
        
        return {
            'task': 'MMLU Test',
            'overall_accuracy': round(overall_accuracy, 2),
            'total_correct': correct_answers,
            'total_questions': total_questions,
            'subject_results': subject_results
        }

    def calculate_perplexity(self, text: str) -> float:
        """Tính perplexity cho một đoạn text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
        
        return perplexity

    def run_perplexity_test(self) -> Dict:
        """Chạy Perplexity test trên nhiều đoạn text khác nhau."""
        print("  📊 Running Perplexity Test...")
        
        perplexities = []
        passage_results = []
        
        for i, passage in enumerate(PERPLEXITY_PASSAGES):
            print(f"    📝 Testing passage {i+1}/{len(PERPLEXITY_PASSAGES)}")
            
            start_time = time.time()
            perplexity = self.calculate_perplexity(passage)
            calc_time = time.time() - start_time
            
            perplexities.append(perplexity)
            passage_results.append({
                'passage_id': i + 1,
                'perplexity': round(perplexity, 4),
                'calculation_time': round(calc_time, 3),
                'text_preview': passage[:100] + "..." if len(passage) > 100 else passage
            })
            
            print(f"      Perplexity: {perplexity:.4f} (calculated in {calc_time:.3f}s)")
        
        avg_perplexity = np.mean(perplexities)
        std_perplexity = np.std(perplexities)
        
        return {
            'task': 'Perplexity Test',
            'average_perplexity': round(avg_perplexity, 4),
            'std_perplexity': round(std_perplexity, 4),
            'min_perplexity': round(min(perplexities), 4),
            'max_perplexity': round(max(perplexities), 4),
            'passage_results': passage_results
        }
        
    def _run_single_benchmark(self, task_name: str, prompts: List[str]) -> Dict:
        print(f"  📊 Benchmarking: {task_name}")
        results = []
        for i, p in enumerate(prompts):
            results.append(self.generate_response(p, f"{task_name} #{i+1}"))
            time.sleep(1) # Nghỉ giữa các lần chạy để ổn định
        
        avg_time = np.mean([r['generation_time'] for r in results])
        avg_tps = np.mean([r['tokens_per_second'] for r in results])
        
        return {
            'task': task_name,
            'avg_generation_time': round(avg_time, 2),
            'avg_tokens_per_second': round(avg_tps, 2),
            'tests': results
        }

    def run_full_benchmark(self) -> Dict:
        """Chạy toàn bộ bộ benchmark trên model gốc."""
        print("\n🚀 Starting Original Model Comprehensive Benchmark (NO QUANTIZATION)")
        print("=" * 80)
        print(f"🎯 Model Precision: {self.model_precision}")
        print(f"💾 Device: {self.device}")
        
        # === MMLU và Perplexity Tests ===
        print("\n🧪 Running Academic & Technical Evaluations...")
        mmlu_results = self.run_mmlu_test()
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        perplexity_results = self.run_perplexity_test()
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # === Traditional Generation Tasks ===
        print("\n📝 Running Traditional Generation Tasks...")
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
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # === Tổng hợp kết quả ===
        overall_tps = [r['avg_tokens_per_second'] for r in benchmark_results]
        memory_info = self.get_memory_usage()
        
        overall_stats = {
            'avg_token_speed': round(np.mean(overall_tps), 2) if overall_tps else 0,
            'mmlu_accuracy': mmlu_results['overall_accuracy'],
            'avg_perplexity': perplexity_results['average_perplexity'],
            'model_precision': self.model_precision,
            'peak_vram_allocated_gb': round(memory_info['gpu_allocated_gb'], 2),
            'peak_vram_reserved_gb': round(memory_info['gpu_reserved_gb'], 2)
        }
        
        return {
            'model_type': 'original_no_quantization',
            'model_hash': self.model_hash,
            'timestamp': datetime.now().isoformat(),
            'overall_stats': overall_stats,
            'mmlu_results': mmlu_results,
            'perplexity_results': perplexity_results,
            'benchmark_results': benchmark_results
        }

    def save_and_print_summary(self, results: Dict):
        """Lưu và in kết quả benchmark."""
        stats = results['overall_stats']
        print("\n" + "="*80 + "\n📊 ORIGINAL MODEL BENCHMARK RESULTS SUMMARY\n" + "="*80)
        print(f"🎯 Model Type: Original (No Quantization)")
        print(f"🔑 Model Hash: {results['model_hash']}")
        print(f"🎭 Model Precision: {stats['model_precision']}")
        print(f"📈 Average Token Speed (Overall): {stats['avg_token_speed']:.2f} tokens/sec")
        print(f"🧠 MMLU Accuracy: {stats['mmlu_accuracy']:.2f}%")
        print(f"📊 Average Perplexity: {stats['avg_perplexity']:.4f}")
        print(f"💾 Peak VRAM Usage: {stats['peak_vram_allocated_gb']:.2f} GB (Reserved: {stats['peak_vram_reserved_gb']:.2f} GB)")
        
        print("\n🧪 Academic Test Results:")
        print(f"  📚 MMLU: {results['mmlu_results']['total_correct']}/{results['mmlu_results']['total_questions']} correct ({stats['mmlu_accuracy']:.2f}%)")
        
        mmlu_by_subject = results['mmlu_results']['subject_results']
        for subject, data in mmlu_by_subject.items():
            print(f"    - {subject:<20}: {data['accuracy']:>6.2f}%")
        
        print(f"  📊 Perplexity Range: {results['perplexity_results']['min_perplexity']:.4f} - {results['perplexity_results']['max_perplexity']:.4f}")
        
        print("\n📋 Generation Task Performance:")
        for result in results['benchmark_results']:
            vram = result['tests'][0]['vram_allocated_gb'] if result['tests'] else 'N/A'
            print(f"  - {result['task']:<25}: {result['avg_tokens_per_second']:.2f} tok/s @ {vram} GB VRAM")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"original_model_benchmark_{self.model_hash}_{timestamp}.json"
        filepath = Path("benchmark_results") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full detailed results saved to: {filepath}")

# === HÀM MAIN ĐỂ CHẠY ORIGINAL MODEL BENCHMARK ===
def main():
    print("🔍 Original Model Benchmark System (No EWQ Quantization)")
    print("=" * 80)
    
    try:
        model_hash = get_model_hash(MODEL_ID)
        print(f"🔑 Model ID: {MODEL_ID}")
        print(f"🔑 Model Hash: {model_hash}")
        
        model, tokenizer = load_original_model(MODEL_ID)
        
        if model is None or tokenizer is None:
            print("❌ Failed to load original model!")
            return
        
        print("✅ Original model loaded and ready for benchmarking!")
        
        # Khởi tạo benchmark suite
        benchmark_suite = OriginalModelBenchmarkSuite(model, tokenizer, model_hash)
        
        # Chạy toàn bộ benchmark
        start_time = time.time()
        results = benchmark_suite.run_full_benchmark()
        total_time = time.time() - start_time
        
        # Thêm thông tin thời gian tổng thể
        results['total_benchmark_time'] = round(total_time, 2)
        
        # Lưu và in kết quả
        benchmark_suite.save_and_print_summary(results)
        
        print(f"\n⏱️ Total benchmark time: {total_time:.2f} seconds")
        print("\n🎉 Original model benchmark completed successfully!")
        
        # Dọn dẹp bộ nhớ
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user.")
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Dọn dẹp cuối cùng
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 Memory cleanup completed.")


if __name__ == "__main__":
    main()