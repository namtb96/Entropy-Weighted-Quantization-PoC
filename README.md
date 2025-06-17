*This document is also available in [English](./README.en.md).*

---
# Lượng tử hóa Trọng số dựa trên Entropy (EWQ) cho Mô hình Ngôn ngữ Lớn

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.04704v2-b31b1b.svg)](https://arxiv.org/html/2503.04704v2)

## 1. Giới thiệu

Dự án này là một bản hiện thực hóa và kiểm thử sâu rộng cho phương pháp **Lượng tử hóa Trọng số dựa trên Entropy (Entropy-based Weight Quantization - EWQ)**, lấy cảm hứng từ ý tưởng được đề xuất trong bài báo khoa học [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2).

**Ý tưởng cốt lõi**: Không phải tất cả các layer trong một mô hình ngôn ngữ lớn (LLM) đều có tầm quan trọng như nhau. Giả thuyết đặt ra là những layer có **entropy thông tin thấp** (phân bố trọng số dễ đoán hơn) có thể được lượng tử hóa ở mức độ sâu (ví dụ: 4-bit) mà ít ảnh hưởng đến chất lượng. Ngược lại, những layer có **entropy cao** (chứa nhiều thông tin phức tạp và quan trọng hơn) cần được giữ ở độ chính xác cao (8-bit hoặc giữ nguyên) để bảo toàn hiệu năng của mô hình.

Dự án này bao gồm:
1.  Một script để phân tích entropy của bất kỳ mô hình transformer nào và **tạo ra một "kế hoạch lượng tử hóa" tùy chỉnh**.
2.  Một cơ chế để áp dụng kế hoạch này, sử dụng `bitsandbytes` để tạo ra một mô hình lượng tử hóa với độ chính xác hỗn hợp (mixed-precision).
3.  Một **bộ kiểm thử (benchmark suite) toàn diện** để đánh giá định lượng hiệu quả của phương pháp EWQ so với mô hình gốc (FP16) và các phương pháp lượng tử hóa tiêu chuẩn công nghiệp như GGUF.

## 2. Cách thức hoạt động

Quy trình được chia thành 3 bước chính, được tự động hóa bằng các script:

### Bước 1: Phân tích Entropy & Tạo kế hoạch (`create_quantization_plan.py`)
- Script tải mô hình gốc lên CPU (để tiết kiệm VRAM).
- Nó lặp qua từng khối (block/layer) của mô hình và tính toán entropy trung bình của các trọng số bên trong.
- Dựa trên phân bố entropy của tất cả các khối, script tính toán giá trị trung bình (`mean`) và độ lệch chuẩn (`std`).
- Một "kế hoạch lượng tử hóa" (`.json`) được tạo ra dựa trên siêu tham số `ENTROPY_THRESHOLD_FACTOR`:
    - **`entropy >= mean`**: Layer quan trọng, giữ nguyên độ chính xác (`raw`).
    - **`mean > entropy >= mean - factor * std`**: Layer có thể lượng tử hóa, sử dụng `8-bit`.
    - **`entropy < mean - factor * std`**: Layer ít nhạy cảm, sử dụng `4-bit`.

### Bước 2: Lượng tử hóa theo kế hoạch (`benchmark_ewq.py`)
- Script này đọc file kế hoạch `.json` đã được tạo.
- Nó tải mô hình gốc lên CPU, sau đó áp dụng kế hoạch lượng tử hóa bằng cách thay thế các layer `nn.Linear` tương ứng bằng `bnb.Linear8bitLt` hoặc `bnb.Linear4bit` của `bitsandbytes`.
- Cuối cùng, mô hình đã được lượng tử hóa theo chiến lược hỗn hợp sẽ được triển khai lên GPU để benchmark.

### Bước 3: Kiểm thử toàn diện (`suite.py`, `tasks.py`, ...)
- Một bộ kiểm thử mạnh mẽ được sử dụng để đánh giá các mô hình trên nhiều phương diện:
    - **Khả năng suy luận & Kiến thức**: Điểm MMLU.
    - **Độ trôi chảy & Tự nhiên của ngôn ngữ**: Điểm Perplexity trên nhiều lĩnh vực (văn chương, khoa học, code, ...).
    - **Hiệu suất**: Tốc độ sinh token (tokens/sec) và mức sử dụng VRAM (GB).
- Kết quả được so sánh với mô hình gốc và các phiên bản GGUF (Q4 và Q8) để có một cái nhìn toàn cảnh.

## 3. Phân tích chi tiết kết quả

Chúng tôi đã thực hiện benchmark trên mô hình `Qwen/Qwen3-8B` với các giá trị `ENTROPY_THRESHOLD_FACTOR` khác nhau. Dưới đây là bảng tổng hợp kết quả so sánh.

| Phiên bản Model | VRAM (GB) | Tốc độ (tok/s) | MMLU (%) | Perplexity |
| :--- | :---: | :---: | :---: | :---: |
| Gốc (FP16) | 15.26 | 47.02 | 69.98 | 26.88 |
| EWQ (Factor 1.0) | 12.05 | 45.91 | 69.45 | 34.70 |
| EWQ (Factor 0.8) | 11.79 | 45.59 | 69.58 | **30.22** |
| **EWQ (Factor 0.5-0.65)** | **11.53** | **~49.0** | **70.50** | ~31.01 |
| GGUF Q4_K_M | **5.59** | 124.47 | 69.32 | 30.07 |
| **GGUF Q8_0** | 8.73 | 86.55 | 70.27 | **26.07** |

*(**Tô đậm**: giá trị tốt nhất trong cột, hoặc kết quả đáng chú ý nhất)*

### Các phát hiện chính:
1.  **Thành công trong việc giảm tài nguyên**: Tất cả các phiên bản EWQ đều giảm đáng kể VRAM sử dụng (khoảng 25%) so với bản gốc.
2.  **MMLU vượt trội**: Bất ngờ lớn nhất là các phiên bản EWQ với `factor` thấp (0.5 - 0.65) không chỉ bảo toàn mà còn **vượt qua MMLU của cả mô hình gốc và GGUF Q8**. Điều này cho thấy việc lượng tử hóa có chọn lọc có thể hoạt động như một hình thức "regularization", giúp mô hình tập trung vào các đặc trưng suy luận quan trọng hơn.
3.  **Tốc độ của GGUF**: Sự vượt trội về tốc độ của các phiên bản GGUF đến từ engine `llama.cpp` được tối ưu hóa ở mức độ thấp (C++), trong khi các thử nghiệm EWQ chạy trên nền tảng `transformers` (Python) nên không thể so sánh trực tiếp về mặt này. Đáng chú ý, các phiên bản EWQ có tốc độ inference nhanh hơn một chút so với bản gốc.
4.  **Sự đánh đổi MMLU vs. Perplexity**: Có một sự đánh đổi thú vị được phát hiện:
    - **`Factor = 0.8`** cho điểm Perplexity tốt nhất (30.22), làm cho nó trở thành lựa chọn tối ưu cho việc **sinh văn bản mượt mà, tự nhiên**.
    - **`Factor = 0.5-0.65`** cho điểm MMLU cao nhất (70.50%), làm cho nó trở thành lựa chọn tối ưu cho các tác vụ **suy luận và hỏi-đáp chính xác**.

## 4. Đánh giá

### Điểm mạnh (Strengths)
*   **Chất lượng suy luận vượt trội**: EWQ đã chứng minh khả năng tạo ra một mô hình "thông minh" hơn cả bản gốc về mặt MMLU.
*   **Hiệu quả về bộ nhớ**: Giảm ~25% VRAM là một con số rất đáng kể, giúp chạy các mô hình lớn trên các GPU có dung lượng hạn chế.
*   **Khả năng tinh chỉnh cao**: Siêu tham số `ENTROPY_THRESHOLD_FACTOR` là một "cần gạt" mạnh mẽ, cho phép người dùng tùy chỉnh sự cân bằng giữa chất lượng suy luận và độ tự nhiên của ngôn ngữ để phù hợp với ứng dụng cụ thể.
*   **Tính tổng quát**: Phương pháp này có thể được áp dụng cho bất kỳ mô hình nào trong hệ sinh thái Hugging Face Transformers.

### Điểm yếu & Sự đánh đổi (Weaknesses & Trade-offs)
*   **Tốc độ Inference**: Do chạy trên nền tảng Python, tốc độ không thể cạnh tranh với các engine được tối ưu hóa bằng C++ như `llama.cpp`.
*   **Sự đánh đổi về chất lượng**: Người dùng cần phải quyết định ưu tiên giữa khả năng suy luận (MMLU) hay khả năng sinh văn bản tự nhiên (Perplexity) để chọn `factor` phù hợp.

## 5. Định hướng tương lai
1.  **Phân tích sâu kế hoạch lượng tử hóa**: So sánh các file `quant_plan.json` được tạo ra bởi các `factor` khác nhau để xác định chính xác những layer nào đang bị thay đổi mức lượng tử hóa, từ đó hiểu rõ hơn nguyên nhân của sự đánh đổi MMLU/Perplexity.
2.  **Đóng gói EWQ sang định dạng GGUF**: Đây là mục tiêu cuối cùng đầy tham vọng. Bằng cách sửa đổi script `convert.py` của `llama.cpp` để đọc và áp dụng `quant_plan.json` của chúng ta, chúng ta có thể tạo ra một file `.gguf` tùy chỉnh. Điều này sẽ kết hợp chiến lược lượng tử hóa thông minh của EWQ và tốc độ inference siêu nhanh của `llama.cpp` để tạo ra một model tối ưu về mọi mặt.
3.  **Kiểm thử trên các kiến trúc khác**: Áp dụng phương pháp EWQ cho các họ model khác (ví dụ: Llama, Mistral, Gemma) để kiểm tra xem các phát hiện này có mang tính phổ quát hay không.

## 6. Hướng dẫn sử dụng

### Bước 1: Cài đặt môi trường
```bash
git clone https://github.com/namtb96/Entropy-Weighted-Quantization-PoC
cd Entropy-Weighted-Quantization-PoC
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
pip install -r requirements.txt
```

### Bước 2: Tạo kế hoạch lượng tử hóa
Chạy script để phân tích mô hình và tạo kế hoạch. Thay đổi MODEL_ID và ENTROPY_THRESHOLD_FACTOR trong script nếu cần.
```bash
python create_quantization_plan.py
```
Thao tác này sẽ tạo một file quant_plan_xxxxxxxx.json trong thư mục quantized_models.

### Bước 3: Chạy Benchmark
Bạn có thể chạy các bài kiểm thử cho từng phiên bản:
a. Chạy benchmark cho phiên bản EWQ:
(Script sẽ tự động tìm file kế hoạch dựa trên cấu hình)
```bash
python benchmark_ewq.py
```
b. Chạy benchmark cho phiên bản gốc (FP16):
```bash
python benchmark_original.py
```
c. Chạy benchmark cho phiên bản GGUF:
(Thay đổi MODEL_REPO_ID và MODEL_FILE trong script cho phù hợp)
```bash
python benchmark_gguf_q4.py
python benchmark_gguf_q8.py
```
Kết quả của mỗi lần chạy sẽ được lưu dưới dạng file .json trong thư mục benchmark_results.

---

---

# Entropy-based Weight Quantization (EWQ) for Large Language Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.04704v2-b31b1b.svg)](https://arxiv.org/html/2503.04704v2)

## 1. Introduction

This project is an implementation and in-depth benchmark of the **Entropy-based Weight Quantization (EWQ)** method, inspired by the concept proposed in the scientific paper [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2).

**The Core Idea**: Not all layers in a Large Language Model (LLM) are equally important. The hypothesis is that layers with **low information entropy** (i.e., more predictable weight distributions) can be quantized more aggressively (e.g., to 4-bit) with minimal impact on quality. Conversely, layers with **high entropy** (containing more complex and critical information) should be kept at a higher precision (8-bit or full precision) to preserve the model's performance.

This project includes:
1.  A script to analyze the entropy of any Transformer-based model and **generate a custom quantization plan**.
2.  A mechanism to apply this plan, creating a mixed-precision quantized model using `bitsandbytes`.
3.  A **comprehensive benchmark suite** to quantitatively evaluate the effectiveness of the EWQ method against the base model (FP16) and industry-standard quantization methods like GGUF.

## 2. How It Works

The process is divided into three main steps, automated by scripts:

### Step 1: Entropy Analysis & Plan Generation (`create_quantization_plan.py`)
- The script loads the base model onto the CPU (to conserve VRAM).
- It iterates through each block (or layer) of the model and calculates the average entropy of its weights.
- Based on the entropy distribution across all blocks, the script computes the mean and standard deviation (`std`).
- A quantization plan (`.json`) is generated based on the `ENTROPY_THRESHOLD_FACTOR` hyperparameter:
    - **`entropy >= mean`**: A critical layer, kept at its original precision (`raw`).
    - **`mean > entropy >= mean - factor * std`**: A quantizable layer, using `8-bit`.
    - **`entropy < mean - factor * std`**: A less sensitive layer, using `4-bit`.

### Step 2: Plan-based Quantization (`benchmark_ewq.py`)
- This script reads the generated `.json` plan file.
- It loads the base model onto the CPU, then applies the quantization plan by replacing the corresponding `nn.Linear` layers with `bitsandbytes`' `bnb.Linear8bitLt` or `bnb.Linear4bit`.
- Finally, the mixed-precision quantized model is deployed to the GPU for benchmarking.

### Step 3: Comprehensive Benchmarking (`suite.py`, `tasks.py`, ...)
- A robust benchmark suite is used to evaluate models across multiple dimensions:
    - **Reasoning & Knowledge**: MMLU score.
    - **Fluency & Language Coherence**: Perplexity score across various domains (literature, science, code, etc.).
    - **Performance**: Token generation speed (tokens/sec) and VRAM usage (GB).
- The results are compared against the base model and GGUF versions (Q4 & Q8) for a holistic view.

## 3. Detailed Results Analysis

We performed benchmarks on the `Qwen/Qwen3-8B` model with various `ENTROPY_THRESHOLD_FACTOR` values. Below is a summary of the comparison.

| Model Version | VRAM (GB) | Speed (tok/s) | MMLU (%) | Perplexity |
| :--- | :---: | :---: | :---: | :---: |
| Base (FP16) | 15.26 | 47.02 | 69.98 | 26.88 |
| EWQ (Factor 1.0) | 12.05 | 45.91 | 69.45 | 34.70 |
| EWQ (Factor 0.8) | 11.79 | 45.59 | 69.58 | **30.22** |
| **EWQ (Factor 0.5-0.65)** | **11.53** | **~49.0** | **70.50** | ~31.01 |
| GGUF Q4_K_M | **5.59** | **124.47** | 69.32 | 30.07 |
| **GGUF Q8_0** | 8.73 | 86.55 | 70.27 | **26.07** |

*(**Bold**: Best value in the column, or the most notable result)*

### Key Findings:
1.  **Successful Resource Reduction**: All EWQ versions significantly reduce VRAM usage (by ~25%) compared to the base model.
2.  **Superior MMLU Performance**: The most surprising finding is that EWQ versions with a low `factor` (0.5 - 0.65) not only preserve but **surpass the MMLU score of both the base model and GGUF Q8**. This suggests that selective quantization can act as a form of regularization, forcing the model to focus on more critical reasoning features.
3.  **GGUF's Speed Advantage**: The superior speed of the GGUF versions comes from the low-level optimized `llama.cpp` (C++) engine. The EWQ tests, running on the `transformers` (Python) framework, are not directly comparable in this regard. Notably, the EWQ versions showed a slight inference speed-up over the base model.
4.  **The MMLU vs. Perplexity Trade-off**: An interesting trade-off was discovered:
    - **`Factor = 0.8`** yielded the best Perplexity score (30.22), making it the optimal choice for **generating smooth, natural text**.
    - **`Factor = 0.5-0.65`** achieved the highest MMLU score (70.50%), making it the optimal choice for **accurate reasoning and question-answering** tasks.

## 4. Evaluation

### Strengths
*   **Superior Reasoning Quality**: EWQ has proven its ability to produce a model that is "smarter" than the original in terms of MMLU score.
*   **Memory Efficiency**: A ~25% reduction in VRAM is a significant achievement, enabling larger models to run on consumer-grade GPUs.
*   **Highly Tunable**: The `ENTROPY_THRESHOLD_FACTOR` hyperparameter acts as a powerful knob, allowing users to customize the trade-off between reasoning quality and language fluency to fit their specific application.
*   **Broad Applicability**: This method can be applied to any model within the Hugging Face Transformers ecosystem.

### Weaknesses & Trade-offs
*   **Inference Speed**: Being based on a Python framework, the speed cannot compete with C++ optimized engines like `llama.cpp`.
*   **Quality Trade-off**: Users must decide whether to prioritize reasoning ability (MMLU) or natural language generation (Perplexity) to select the appropriate `factor`.

## 5. Future Directions
1.  **In-depth Analysis of Quantization Plans**: Compare the `quant_plan.json` files generated by different `factor` values to identify exactly which layers are having their precision levels changed. This will help to better understand the cause of the MMLU/Perplexity trade-off.
2.  **Packaging EWQ into GGUF Format**: This is the ambitious ultimate goal. By modifying the `convert.py` script from `llama.cpp` to read and apply our `quant_plan.json`, we could create a custom `.gguf` file. This would combine the intelligent quantization strategy of EWQ with the blazing-fast inference speed of `llama.cpp`.
3.  **Testing on Other Architectures**: Apply the EWQ method to other model families (e.g., Llama, Mistral, Gemma) to test whether these findings are universal.

## 6. Usage Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/namtb96/Entropy-Weighted-Quantization-PoC
cd Entropy-Weighted-Quantization-PoC
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
pip install -r requirements.txt
```

### Step 2: Generate Quantization Plan
Run the script to analyze the model and generate a plan. Modify `MODEL_ID` and `ENTROPY_THRESHOLD_FACTOR` in the script if necessary.
```bash
python create_quantization_plan.py
```
This will create a quant_plan_xxxxxxxx.json file in the quantized_models directory.

### Step 3: Run the Benchmarks
You can run the benchmark tests for each version:
a. Run the benchmark for the EWQ version:
(The script will automatically find the plan file based on the configuration)
```bash
python benchmark_ewq.py
```

b. Run the benchmark for the original (FP16) version:
```bash
python benchmark_original.py
```

c. Run the benchmark for GGUF versions:
(Modify MODEL_REPO_ID and MODEL_FILE in the scripts accordingly)
```bash
python benchmark_gguf_q4.py
python benchmark_gguf_q8.py
```
The results of each run will be saved as a .json file in the benchmark_results directory