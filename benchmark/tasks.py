import torch
import numpy as np
import csv
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

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

def run_mmlu_test(model, tokenizer, device, mmlu_dir: Path) -> Dict:
    print("  🧠 Running MMLU Test from CSV files...")
    mmlu_data = _load_mmlu_data_from_csv(mmlu_dir)
    # ... (toàn bộ logic của run_mmlu_test giống hệt phiên bản trước, nhưng nhận model, tokenizer, device làm tham số) ...
    if not mmlu_data:
        return {'task': 'MMLU Test', 'status': 'Skipped', 'reason': 'No MMLU data found or loaded.'}
    
    correct_answers, total_questions, subject_results = 0, 0, {}
    
    for subject, questions in mmlu_data.items():
        print(f"    📚 Testing subject: {subject} ({len(questions)} questions)")
        subject_correct = 0
        for q in tqdm(questions, desc=f"  -> {subject}", leave=False):
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(q['choices'])])
            prompt = f"Question: {q['question']}\n{choices_text}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.1)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
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

def run_perplexity_test(model, tokenizer, device, categories: Dict[str, list]) -> Dict:
    """
    Chạy Perplexity test trên các danh mục văn bản khác nhau.
    
    Args:
        model: Model ngôn ngữ đã được tải.
        tokenizer: Tokenizer tương ứng với model.
        device: Thiết bị để chạy tính toán (ví dụ: 'cuda' hoặc 'cpu').
        categories: Một từ điển trong đó key là tên danh mục (str) và 
                    value là một danh sách các đoạn văn (list[str]).

    Returns:
        Một từ điển chứa kết quả chi tiết theo từng danh mục và kết quả tổng thể.
    """
    print("  📊 Running Perplexity Test across multiple categories...")
    
    category_results = {}
    all_perplexities = []
    
    # Vòng lặp ngoài: lặp qua từng danh mục (ví dụ: 'general_knowledge', 'code_python')
    for category_name, passages in categories.items():
        print(f"    -> Testing category: '{category_name}' ({len(passages)} passages)")
        
        perplexities_for_this_category = []
        
        # Vòng lặp trong: lặp qua từng đoạn văn trong danh mục hiện tại
        for passage in tqdm(passages, desc=f"      {category_name}", leave=False):
            # Bỏ qua các đoạn văn trống để tránh lỗi
            if not passage.strip():
                continue

            try:
                inputs = tokenizer(passage, return_tensors="pt", truncation=True, max_length=512).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    
                    # Guard-clause để tránh lỗi math domain error nếu loss không dương
                    if loss > 0:
                        perplexity = torch.exp(torch.tensor(loss)).item()
                        perplexities_for_this_category.append(perplexity)

            except Exception as e:
                print(f"      [Warning] Could not calculate perplexity for a passage in '{category_name}': {e}")

        # Tính toán kết quả cho danh mục hiện tại
        if perplexities_for_this_category:
            avg_ppl = np.mean(perplexities_for_this_category)
            category_results[category_name] = {
                'average_perplexity': round(avg_ppl, 4),
                'num_passages': len(perplexities_for_this_category) # Chỉ đếm những passage đã tính được
            }
            # Thêm tất cả điểm PPL của danh mục này vào danh sách tổng
            all_perplexities.extend(perplexities_for_this_category)
            print(f"    📊 Category '{category_name}' Average PPL: {avg_ppl:.4f}")
        else:
            print(f"    ⚠️ No valid perplexity scores calculated for category '{category_name}'.")

    # Tính toán kết quả tổng thể từ tất cả các điểm PPL đã thu thập
    overall_avg_ppl = np.mean(all_perplexities) if all_perplexities else 0.0
    
    print(f"\n    📈 Overall Average Perplexity across all categories: {overall_avg_ppl:.4f}")

    return {
        'task': 'Perplexity Test',
        'overall_average_perplexity': round(overall_avg_ppl, 4),
        'category_results': category_results
    }