import numpy as np
import csv
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
from llama_cpp import Llama
from datasets import load_dataset
from evaluate import load 

def _load_mmlu_data_from_csv(mmlu_dir: Path, max_questions_per_subject: Optional[int] = None) -> Dict:
    print(f"  🔍 Searching for MMLU test files in: {mmlu_dir}")
    if not mmlu_dir.is_dir():
        print(f"  ⚠️ Warning: MMLU directory not found at '{mmlu_dir}'. Skipping MMLU test.")
        return {}
    all_questions = {}
    csv_files = sorted(list(mmlu_dir.glob("*_test.csv")))
    if not csv_files:
        print(f"  ⚠️ Warning: No '*_test.csv' files found in '{mmlu_dir}'. Skipping MMLU test.")
        return {}
    for csv_file in csv_files:
        subject_name = csv_file.name.replace("_test.csv", "")
        subject_questions = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 6: continue
                    question_text, choice_a, choice_b, choice_c, choice_d, answer_letter = row
                    answer_index = ord(answer_letter.strip().upper()) - ord('A')
                    q_data = {"question": question_text, "choices": [choice_a, choice_b, choice_c, choice_d], "answer": answer_index}
                    subject_questions.append(q_data)
            if max_questions_per_subject:
                all_questions[subject_name] = subject_questions[:max_questions_per_subject]
            else:
                all_questions[subject_name] = subject_questions
        except Exception as e:
            print(f"    ❌ Error reading file {csv_file.name}: {e}")
    return all_questions

def run_mmlu_test(llm: Llama, mmlu_dir: Path) -> Dict:
    print("  🧠 Running MMLU Test for GGUF model...")
    mmlu_data = _load_mmlu_data_from_csv(mmlu_dir)
    if not mmlu_data:
        return {'task': 'MMLU Test', 'status': 'Skipped', 'reason': 'No MMLU data found or loaded.'}
    
    correct_answers, total_questions, subject_results = 0, 0, {}
    
    for subject, questions in mmlu_data.items():
        print(f"    📚 Testing subject: {subject} ({len(questions)} questions)")
        subject_correct = 0
        for q in tqdm(questions, desc=f"  -> {subject}", leave=False):
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(q['choices'])])
            # Prompt đơn giản để model chỉ trả về ký tự A, B, C, hoặc D
            prompt = f"Question: {q['question']}\n{choices_text}\nAnswer:"
            
            completion = llm.create_completion(
                prompt,
                max_tokens=5, # Chỉ cần một token là đủ
                temperature=0.0 # Lấy câu trả lời chắc chắn nhất
            )
            response = completion['choices'][0]['text'].strip()
            
            predicted_letter = response[0].upper() if response and response[0].upper() in 'ABCD' else 'X'
            correct_letter = chr(65 + q['answer'])
            if predicted_letter == correct_letter:
                subject_correct += 1
        
        correct_answers += subject_correct
        total_questions += len(questions)
        subject_accuracy = subject_correct / len(questions) * 100 if questions else 0
        subject_results[subject] = {'correct': subject_correct, 'total': len(questions), 'accuracy': round(subject_accuracy, 2)}
        print(f"    📊 {subject} accuracy: {subject_accuracy:.2f}% ({subject_correct}/{len(questions)})")
    
    overall_accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0
    return {'task': 'MMLU Test', 'overall_accuracy': round(overall_accuracy, 2), 'total_correct': correct_answers, 'total_questions': total_questions}

def _load_generation_test_data(dataset_name: str, split: str, prompt_column: str, reference_column: str, num_samples: int, cache_dir: str) -> List[Dict[str, str]]:
    """
    Tải và xử lý dữ liệu từ một dataset trên Hugging Face Hub, sử dụng cache.
    """
    print(f"\n  📦 Loading {num_samples} samples from Hugging Face dataset '{dataset_name}'...")
    print(f"  💾 Cache directory: '{Path(cache_dir).resolve()}'")
    try:
        dataset = load_dataset(dataset_name, split=f"{split}[:{num_samples}]", cache_dir=cache_dir)
    except Exception as e:
        print(f"  ❌ Failed to load dataset '{dataset_name}'. Error: {e}")
        return []

    formatted_data = []
    for sample in dataset:
        if prompt_column not in sample or reference_column not in sample:
            continue
        prompt = f"Summarize the following text:\n\n{sample[prompt_column]}"
        reference = sample[reference_column]
        formatted_data.append({'prompt': prompt, 'reference': reference})
    
    return formatted_data

def run_bleu_rouge_test(llm: Llama, dataset_name: str, split: str, prompt_column: str, reference_column: str, num_samples: int = 50) -> Optional[Dict]:
    """
    Chạy kiểm tra chất lượng sinh văn bản bằng cách tải dữ liệu, tạo dự đoán và tính điểm BLEU & ROUGE.
    Dữ liệu sẽ được cache vào thư mục './generation_test_cache'.
    """
    print("\n  ✍️  Running BLEU & ROUGE Generation Quality Test...")
    
    # Định nghĩa thư mục cache
    cache_directory = "./generation_test_cache"

    # Tải dữ liệu bên trong hàm
    test_data = _load_generation_test_data(
        dataset_name=dataset_name,
        split=split,
        prompt_column=prompt_column,
        reference_column=reference_column,
        num_samples=num_samples,
        cache_dir=cache_directory
    )

    if not test_data:
        print("  ⚠️ Skipping test as no data was loaded.")
        return {'task': 'BLEU/ROUGE Test', 'status': 'Skipped', 'reason': 'Failed to load test data.'}

    try:
        bleu_metric = load('bleu')
        rouge_metric = load('rouge')
    except Exception as e:
        print(f"  ❌ Error loading metrics: {e}")
        return {'task': 'BLEU/ROUGE Test', 'status': 'Failed', 'reason': str(e)}

    predictions = []
    references = []

    print(f"  🚀 Generating {len(test_data)} predictions from the model...")
    for item in tqdm(test_data, desc="  -> Generating", leave=False):
        prompt = item['prompt']
        reference_text = item['reference']
        
        completion = llm.create_completion(prompt, max_tokens=256, temperature=0.1, stop=["\n\n", "Summarize the following text:"])
        prediction_text = completion['choices'][0]['text'].strip()
        
        predictions.append(prediction_text)
        references.append(reference_text)

    print("  🔢 Calculating scores...")
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    final_results = {
        'bleu': round(bleu_score['bleu'], 4),
        'rouge1': round(rouge_score['rouge1'], 4),
        'rouge2': round(rouge_score['rouge2'], 4),
        'rougeL': round(rouge_score['rougeL'], 4),
        'dataset': dataset_name,
        'num_samples': len(predictions)
    }
    
    print(f"    📊 BLEU Score: {final_results['bleu']:.4f}")
    print(f"    📊 ROUGE-L Score: {final_results['rougeL']:.4f}")
    
    return {'task': 'BLEU/ROUGE Test', 'results': final_results}