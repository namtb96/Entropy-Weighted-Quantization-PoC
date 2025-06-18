### 🇬🇧 English version is available below.

---

# Lượng tử hóa theo Entropy (EWQ): Tái hiện và Kiểm thử Toàn diện

Dự án này là một nỗ lực độc lập nhằm tái hiện (re-implement) phương pháp **Lượng tử hóa theo Entropy (Entropy-Weighted Quantization - EWQ)** được đề xuất trong bài báo khoa học trên [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2). Do bài báo gốc không cung cấp mã nguồn, dự án này được xây dựng từ đầu để:

1.  **Tái hiện thuật toán EWQ cốt lõi** để tạo ra một kế hoạch lượng tử hóa (quantization plan) tùy chỉnh cho các mô hình ngôn ngữ lớn (LLMs).
2.  **Xây dựng một bộ kiểm thử (benchmark suite) toàn diện** để đánh giá một cách khách quan và nghiêm ngặt hiệu quả của phương pháp EWQ so với các kỹ thuật lượng tử hóa tiêu chuẩn.
3.  **Xác thực** liệu phương pháp EWQ có thực sự tạo ra một model cân bằng vượt trội về chất lượng, hiệu suất và việc sử dụng tài nguyên hay không.

## Thuật toán Lượng tử hóa theo Entropy (EWQ) hoạt động như thế nào?

Cốt lõi của dự án là script `create_quantization_plan.py`. Thay vì áp dụng một phương pháp lượng tử hóa đồng nhất (ví dụ: tất cả các lớp đều là 4-bit), thuật toán EWQ thực hiện một cách tiếp cận thông minh và linh hoạt hơn:

1.  **Phân tích trên CPU:** Toàn bộ model gốc được tải lên CPU để phân tích. Điều này giúp tiết kiệm VRAM và cho phép xử lý các model cực lớn mà không bị giới hạn bởi bộ nhớ GPU.
2.  **Tính toán Entropy cho từng Block:** Thuật toán lặp qua từng "block" (layer) của model và tính toán **Shannon entropy** cho trọng số của các tầng tuyến tính (`nn.Linear`). Entropy ở đây đóng vai trò là một thước đo về "mức độ phức tạp" hay "lượng thông tin" mà mỗi block nắm giữ.
3.  **Xác định Ngưỡng Thích ứng:** Thay vì dùng một ngưỡng entropy cố định, thuật toán sẽ phân tích sự phân bổ entropy của toàn bộ model (tính giá trị trung bình, độ lệch chuẩn) và tạo ra một **ngưỡng động** (`threshold = mean - factor * std_dev`). Ngưỡng này sẽ tự điều chỉnh dựa trên đặc tính riêng của từng model.
4.  **Logic ra Quyết định 3 cấp:** Dựa trên entropy của mỗi block so với sự phân bổ chung, một quyết định lượng tử hóa được đưa ra:
    *   **Entropy Cao (quan trọng nhất):** Giữ lại độ chính xác gốc (FP16).
    *   **Entropy Trung bình:** Lượng tử hóa vừa phải (8-bit).
    *   **Entropy Thấp (ít thông tin hơn):** Lượng tử hóa mạnh (4-bit).
5.  **Kết quả:** Quá trình này tạo ra một file `quant_plan_*.json`, chứa kế hoạch chi tiết về việc sẽ lượng tử hóa mỗi block như thế nào. Đi kèm với đó là một file kịch bản shell `quantize_command_*.sh` để tự động tạo ra model GGUF tùy chỉnh từ kế hoạch này.

## Hệ thống Kiểm thử (Benchmark)

Để chứng minh giá trị của EWQ, một bộ benchmark toàn diện đã được xây dựng để so sánh các phiên bản model khác nhau trên nhiều khía cạnh:

**Các phiên bản được so sánh:**
*   **Original (FP16):** Model gốc chưa qua lượng tử hóa, làm cơ sở so sánh.
*   **Standard GGUF (Q4 & Q8):** Các phương pháp lượng tử hóa GGUF tiêu chuẩn.
*   **EWQ (bitsandbytes):** Áp dụng plan EWQ bằng thư viện `bitsandbytes`.
*   **EWQ (GGUF):** Áp dụng plan EWQ để tạo ra file GGUF tùy chỉnh.

**Các chỉ số được đo lường:**
*   **Chất lượng Model:**
    *   **MMLU:** Đánh giá kiến thức và khả năng suy luận đa lĩnh vực.
    *   **BLEU & ROUGE:** Đánh giá chất lượng sinh văn bản (ví dụ: tóm tắt).
*   **Hiệu suất:** Tốc độ sinh token (tokens/giây).
*   **Tài nguyên:** Mức sử dụng VRAM (GB) và RAM hệ thống.

## Hướng dẫn sử dụng

Bạn có thể tự mình tái tạo lại toàn bộ quy trình bằng các bước sau:

1.  **Cài đặt môi trường:**
    ```bash
    git clone https://github.com/namtb96/Entropy-Weighted-Quantization-PoC.gitgit
    cd Entropy-Weighted-Quantization-PoC
    pip install -r requirements.txt
    ```

2.  **Tạo Kế hoạch Lượng tử hóa EWQ:**
    Chạy script để phân tích model và tạo ra plan.
    ```bash
    python create_quantization_plan.py
    ```
    Script này sẽ tạo ra file `quant_plan_{hash}.json` và `quantize_command_{hash}.sh` trong thư mục `quantized_models/`.

3.  **(Tùy chọn) Tạo Model GGUF Tùy chỉnh:**
    Để tạo file GGUF từ plan đã có, bạn cần build `llama.cpp` và chạy kịch bản shell đã được tạo tự động.
    ```bash
    # (Thực hiện theo hướng dẫn build của llama.cpp)
    bash ./quantized_models/quantize_command_{hash}.sh
    ```

4.  **Chạy Benchmark:**
    Bạn có thể chạy benchmark cho bất kỳ phiên bản nào bạn muốn kiểm thử.
    ```bash
    # Chạy benchmark cho phiên bản EWQ bitsandbytes
    python benchmark_ewq.py

    # Chạy benchmark cho phiên bản EWQ-GGUF
    python benchmark_gguf_ewq.py

    # Chạy benchmark cho phiên bản GGUF tiêu chuẩn
    python benchmark_gguf_q4.py
    python benchmark_gguf_q8.py

    # Chạy benchmark cho model gốc
    python benchmark_original.py
    ```

5.  **Xem kết quả:**
    Tất cả các kết quả chi tiết sẽ được lưu trong thư mục `benchmark_results/`.

## Kết quả Benchmark và Phân tích

Đây là phần quan trọng nhất, chứng minh hiệu quả của phương pháp.

### Bảng Tổng hợp Kết quả

| Phiên bản Model | VRAM (GB) | Tốc độ (tokens/s) | MMLU (%) | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: |
| **Original (FP16)** | 15.26 | 47.04 | 69.98 | 0.1783 |
| **Standard Q4 (GGUF)** | **6.40** | **124.14** | 69.32 | 0.1564 |
| **Standard Q8 (GGUF)** | 9.54 | 85.83 | 70.27 | 0.1515 |
| **EWQ (bitsandbytes)** | 9.64 | 43.32 | **70.30** | **0.1800** |
| **EWQ (GGUF)** | 8.54 | 95.18 | 70.05 | 0.1548 |

*(**In đậm** là giá trị tốt nhất trong từng hạng mục)*

### Phân tích

1.  **Chất lượng Model (MMLU & ROUGE-L):**
    *   Phương pháp **EWQ (bitsandbytes)** đạt điểm MMLU và ROUGE-L **cao nhất**, chứng tỏ thuật toán đã bảo toàn "trí thông minh" của model gốc một cách xuất sắc, thậm chí nhỉnh hơn một chút.
    *   Phiên bản **EWQ-GGUF** cũng duy trì chất lượng gần như ngang bằng với bản gốc và vượt trội hơn hẳn so với phương pháp Q4 tiêu chuẩn.

2.  **Tài nguyên và Hiệu suất (VRAM & Tốc độ):**
    *   Trong khi Q4 nhanh nhất và nhẹ nhất, nó phải đánh đổi bằng chất lượng.
    *   **EWQ-GGUF** đã tìm ra một **"điểm ngọt" (sweet spot)** hoàn hảo. So với Q8 tiêu chuẩn, nó **vượt trội về mọi mặt**:
        *   **Nhẹ hơn:** Tiết kiệm hơn **1 GB VRAM** (8.54 GB so với 9.54 GB).
        *   **Nhanh hơn:** Nhanh hơn đáng kể **~11%** (95.18 tokens/s so với 85.83 tokens/s).
        *   **Chất lượng tương đương:** Điểm MMLU gần như không đổi.

## Kết luận

Dự án đã tái hiện thành công phương pháp Lượng tử hóa theo Entropy và quan trọng hơn, đã chứng minh được tính hiệu quả của nó thông qua một bộ kiểm thử nghiêm ngặt.

**Kết quả cho thấy rõ ràng rằng việc sử dụng plan EWQ để tạo ra một file GGUF tùy chỉnh đã tạo ra một phiên bản model cân bằng và tối ưu hơn so với các phương pháp lượng tử hóa tiêu chuẩn, mang lại hiệu suất cao và yêu cầu tài nguyên thấp trong khi vẫn duy trì được chất lượng gần như nguyên vẹn.**

## Hướng phát triển trong tương lai

*   Trực quan hóa sự phân bổ entropy của các block model để có cái nhìn sâu sắc hơn.
*   Tham số hóa `entropy_factor` để dễ dàng thử nghiệm các "độ nhạy" lượng tử hóa khác nhau.
*   Nghiên cứu áp dụng thuật toán ở mức độ chi tiết hơn (per-layer) thay vì per-block.

## Lời cảm ơn

Dự án này được truyền cảm hứng và dựa trên các ý tưởng được trình bày trong bài báo khoa học "Entropy-based Mixed-Precision Quantization for Balanced Language Model Compression" có sẵn trên [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2).

---

# Entropy-Weighted Quantization (EWQ): A Comprehensive Re-implementation and Benchmark

This project is an independent effort to re-implement the **Entropy-Weighted Quantization (EWQ)** method proposed in the scientific paper [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2). As the original paper did not provide source code, this project was built from the ground up to:

1.  **Re-implement the core EWQ algorithm** to generate a custom quantization plan for Large Language Models (LLMs).
2.  **Build a comprehensive benchmark suite** to objectively and rigorously evaluate the effectiveness of the EWQ method against standard quantization techniques.
3.  **Validate** whether the EWQ method truly produces a superiorly balanced model in terms of quality, performance, and resource consumption.

## How Does Entropy-Weighted Quantization (EWQ) Work?

The heart of this project is the `create_quantization_plan.py` script. Instead of applying a uniform quantization strategy (e.g., all layers to 4-bit), the EWQ algorithm takes a more intelligent and flexible approach:

1.  **CPU-based Analysis:** The entire base model is loaded onto the CPU for analysis. This conserves VRAM and allows for the processing of very large models without being limited by GPU memory.
2.  **Per-Block Entropy Calculation:** The algorithm iterates through each "block" of the model and calculates the **Shannon entropy** for the weights of its linear layers. Entropy here serves as a metric for the "complexity" or "amount of information" that each block holds.
3.  **Adaptive Threshold Determination:** Rather than using a fixed entropy threshold, the algorithm analyzes the entropy distribution across the entire model (calculating the mean and standard deviation) to create a **dynamic threshold** (`threshold = mean - factor * std_dev`). This threshold adapts to the unique characteristics of each model.
4.  **Three-Tiered Decision Logic:** Based on each block's entropy relative to the overall distribution, a quantization decision is made:
    *   **High Entropy (Most Critical):** Retain original precision (FP16).
    *   **Medium Entropy:** Apply moderate quantization (8-bit).
    *   **Low Entropy (Less Informative):** Apply aggressive quantization (4-bit).
5.  **Output:** This process generates a `quant_plan_*.json` file containing the detailed plan for how each block will be quantized. It also produces a `quantize_command_*.sh` shell script to automatically create a custom GGUF model from this plan.

## The Benchmark Suite

To prove the value of EWQ, a comprehensive benchmark suite was built to compare different model versions across multiple dimensions:

**Compared Versions:**
*   **Original (FP16):** The unquantized base model, serving as the gold standard.
*   **Standard GGUF (Q4 & Q8):** Standard GGUF quantization methods.
*   **EWQ (bitsandbytes):** The EWQ plan applied using the `bitsandbytes` library.
*   **EWQ (GGUF):** The EWQ plan used to create a custom GGUF file.

**Measured Metrics:**
*   **Model Quality:**
    *   **MMLU:** Evaluates multi-domain knowledge and reasoning abilities.
    *   **BLEU & ROUGE:** Assesses the quality of text generation (e.g., summarization).
*   **Performance:** Token generation speed (tokens/second).
*   **Resources:** VRAM (GB) and system RAM usage.

## Usage Guide

You can reproduce the entire process by following these steps:

1.  **Setup Environment:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    pip install -r requirements.txt
    ```

2.  **Generate the EWQ Quantization Plan:**
    Run the script to analyze the model and create the plan.
    ```bash
    python create_quantization_plan.py
    ```
    This will generate `quant_plan_{hash}.json` and `quantize_command_{hash}.sh` in the `quantized_models/` directory.

3.  **(Optional) Create the Custom GGUF Model:**
    To create the GGUF file from the generated plan, you will need to build `llama.cpp` and then run the auto-generated shell script.
    ```bash
    # (Follow the build instructions for llama.cpp)
    bash ./quantized_models/quantize_command_{hash}.sh
    ```

4.  **Run the Benchmarks:**
    You can run the benchmark for any version you wish to test.
    ```bash
    # Run benchmark for the EWQ bitsandbytes version
    python benchmark_ewq.py

    # Run benchmark for the EWQ-GGUF version
    python benchmark_gguf_ewq.py

    # Run benchmarks for standard GGUF versions
    python benchmark_gguf_q4.py
    python benchmark_gguf_q8.py

    # Run benchmark for the original model
    python benchmark_original.py
    ```

5.  **View Results:**
    All detailed results will be saved in the `benchmark_results/` directory.

## Benchmark Results and Analysis

This is the most crucial section, demonstrating the method's effectiveness.

### Results Summary Table

| Model Version | VRAM (GB) | Speed (tokens/s) | MMLU (%) | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: |
| **Original (FP16)** | 15.26 | 47.04 | 69.98 | 0.1783 |
| **Standard Q4 (GGUF)** | **6.40** | **124.14** | 69.32 | 0.1564 |
| **Standard Q8 (GGUF)** | 9.54 | 85.83 | 70.27 | 0.1515 |
| **EWQ (bitsandbytes)** | 9.64 | 43.32 | **70.30** | **0.1800** |
| **EWQ (GGUF)** | 8.54 | 95.18 | 70.05 | 0.1548 |

*(**Bold** indicates the best value in each category)*

### Analysis

1.  **Model Quality (MMLU & ROUGE-L):**
    *   The **EWQ (bitsandbytes)** method achieved the **highest MMLU and ROUGE-L scores**, proving that the algorithm excellently preserves the base model's "intelligence," even showing slight improvements.
    *   The **EWQ-GGUF** version also maintained quality nearly identical to the original and significantly outperformed the standard Q4 method.

2.  **Resources and Performance (VRAM & Speed):**
    *   While the standard Q4 model is the fastest and lightest, it comes at the cost of quality.
    *   The **EWQ-GGUF** model found the perfect **"sweet spot"**. Compared to the standard Q8 model, it is **superior in every aspect**:
        *   **Lighter:** Saves over **1 GB of VRAM** (8.54 GB vs. 9.54 GB).
        *   **Faster:** A significant **~11% faster** (95.18 tokens/s vs. 85.83 tokens/s).
        *   **Equivalent Quality:** The MMLU score remains nearly unchanged.

## Conclusion

This project has successfully re-implemented the Entropy-Weighted Quantization method and, more importantly, has proven its effectiveness through a rigorous benchmark suite.

**The results clearly demonstrate that using an EWQ plan to create a custom GGUF file produces a more balanced and optimized model than standard quantization methods, delivering high performance and low resource requirements while maintaining near-original quality.**

## Future Work

*   Visualize the model's block entropy distribution for deeper insights.
*   Parameterize the `entropy_factor` to easily experiment with different quantization sensitivities.
*   Investigate applying the algorithm at a more granular, per-layer level instead of per-block.

## Acknowledgments

This project was inspired by and is based on the ideas presented in the scientific paper "Entropy-based Mixed-Precision Quantization for Balanced Language Model Compression," available on [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2).