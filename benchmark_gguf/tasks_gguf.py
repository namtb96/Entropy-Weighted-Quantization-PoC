import numpy as np
import csv
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
from llama_cpp import Llama
from scipy.special import log_softmax

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
    return {'task': 'MMLU Test', 'overall_accuracy': round(overall_accuracy, 2), 'total_correct': correct_answers, 'total_questions': total_questions, 'subject_results': subject_results}


def run_perplexity_test(llm: Llama, categories: Dict[str, list]) -> Dict:
    """ Chạy Perplexity test bằng cách tính toán thủ công từ logits. """
    print("  📊 Running Perplexity Test for GGUF model (manual calculation)...")
    
    category_results = {}
    all_perplexities = []
    
    for category_name, passages in categories.items():
        print(f"    -> Testing category: '{category_name}' ({len(passages)} passages)")
        perplexities_for_this_category = []
        
        for passage in tqdm(passages, desc=f"      {category_name}", leave=False):
            if not passage.strip(): continue

            try:
                tokens = llm.tokenize(passage.encode("utf-8"))
                
                # Perplexity không xác định cho chuỗi có ít hơn 2 token
                if len(tokens) < 2:
                    continue

                llm.reset() # Đảm bảo context sạch cho mỗi lần tính
                llm.eval(tokens)
                
                # Lấy logits từ model (yêu cầu logits_all=True khi khởi tạo)
                logits = np.array(llm.scores) # Shape: (n_tokens, n_vocab)
                
                # Tính toán cross-entropy loss
                # Chúng ta dự đoán token i dựa trên token 0 đến i-1
                # Vì vậy, logits tại bước i-1 được dùng để dự đoán token i
                # Logits cho token cuối cùng không được sử dụng để dự đoán gì cả.
                shifted_logits = logits[:-1, :]
                
                # Lấy ID của các token thực tế mà chúng ta cần dự đoán
                target_tokens = tokens[1:]
                
                # Tính log-softmax trên logits
                log_probs = log_softmax(shifted_logits, axis=-1)
                
                # Lấy log-probability của token thực tế
                log_likelihoods = log_probs[np.arange(len(target_tokens)), target_tokens]
                
                # Loss là negative log-likelihood trung bình
                nll = -np.sum(log_likelihoods)
                loss = nll / len(target_tokens)
                
                perplexity = np.exp(loss)
                perplexities_for_this_category.append(perplexity)

            except Exception as e:
                print(f"      [Warning] Could not calculate perplexity for a passage in '{category_name}': {e}")
        
        llm.reset() # Reset context sau mỗi category

        if perplexities_for_this_category:
            avg_ppl = np.mean(perplexities_for_this_category)
            category_results[category_name] = {'average_perplexity': round(avg_ppl, 4), 'num_passages': len(perplexities_for_this_category)}
            all_perplexities.extend(perplexities_for_this_category)
            print(f"    📊 Category '{category_name}' Average PPL: {avg_ppl:.4f}")
        else:
            print(f"    ⚠️ No valid perplexity scores calculated for category '{category_name}'.")

    overall_avg_ppl = np.mean(all_perplexities) if all_perplexities else 0.0
    print(f"\n    📈 Overall Average Perplexity across all categories: {overall_avg_ppl:.4f}")

    return {
        'task': 'Perplexity Test',
        'overall_average_perplexity': round(overall_avg_ppl, 4),
        'category_results': category_results
    }