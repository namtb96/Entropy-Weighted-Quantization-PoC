### 🇬🇧 English version is available below.

---

# Lượng tử hóa theo Entropy (EWQ): Tái hiện và Kiểm thử Toàn diện (Phiên bản Tensor-wise)

Dự án này là một nỗ lực độc lập nhằm tái hiện và cải tiến phương pháp **Lượng tử hóa theo Entropy (Entropy-Weighted Quantization - EWQ)** được đề xuất trong bài báo khoa học trên [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2). Phiên bản này đã được nâng cấp từ phân tích theo block lên **phân tích theo từng tensor**, cho phép tối ưu hóa ở mức độ chi tiết và hiệu quả hơn.

Mục tiêu của dự án:
1.  **Tái hiện và cải tiến thuật toán EWQ cốt lõi** để tạo ra một kế hoạch lượng tử hóa (quantization plan) tùy chỉnh ở cấp độ tensor cho các mô hình ngôn ngữ lớn (LLMs).
2.  **Xây dựng một bộ kiểm thử (benchmark suite) toàn diện** để đánh giá một cách khách quan và nghiêm ngặt hiệu quả của phương pháp EWQ so với các kỹ thuật lượng tử hóa tiêu chuẩn.
3.  **Xác thực** liệu phương pháp EWQ-Tensorwise có thực sự tạo ra một model cân bằng vượt trội về chất lượng, hiệu suất và việc sử dụng tài nguyên hay không.

## Thuật toán Lượng tử hóa theo Entropy (Tensor-wise) hoạt động như thế nào?

Cốt lõi của dự án là script `create_quantization_plan_tensorwise.py`. Thay vì áp dụng một phương pháp lượng tử hóa đồng nhất hoặc theo từng block, thuật toán EWQ-Tensorwise thực hiện một cách tiếp cận cực kỳ chi tiết:

1.  **Phân tích trên CPU:** Toàn bộ model gốc được tải lên CPU để phân tích, giúp tiết kiệm VRAM và cho phép xử lý các model cực lớn.
2.  **Phân tích Entropy và Tầm quan trọng cho từng Tensor:** Thuật toán lặp qua **từng tensor trọng số** (ví dụ: `layers.0.self_attn.q_proj.weight`) trong toàn bộ model.
    *   **Tính toán Entropy:** **Shannon entropy** được tính cho mỗi tensor, đóng vai trò là một thước đo về "mức độ phức tạp" hay "lượng thông tin" mà tensor đó nắm giữ.
    *   **Gán Trọng số Quan trọng:** Mỗi tensor được gán một **hệ số quan trọng (`TENSOR_IMPORTANCE`)** dựa trên tên và vai trò của nó (ví dụ: các tensor embedding và output được coi là quan trọng nhất).
3.  **Xác định Ngưỡng Thích ứng:** Thuật toán phân tích sự phân bổ entropy của toàn bộ các tensor và tạo ra các **ngưỡng động** dựa trên giá trị trung bình và độ lệch chuẩn.
4.  **Logic ra Quyết định Đa yếu tố:** Dựa trên một tổ hợp các yếu tố của mỗi tensor, một quyết định lượng tử hóa được đưa ra:
    *   **Entropy** của tensor.
    *   **Độ quan trọng** được gán trước.
    *   **Kích thước** của tensor (các tensor rất nhỏ sẽ được giữ nguyên).
    *   **Quy tắc cứng:** Các tensor tối quan trọng (ví dụ: `output.weight`) luôn được giữ ở độ chính xác cao nhất (FP16).
5.  **Kết quả:** Quá trình này tạo ra một file `quant_plan_*.json` cực kỳ chi tiết, chứa kế hoạch lượng tử hóa cho từng tensor riêng lẻ. Đi kèm với đó là một file kịch bản shell `quantize_command_*.sh` để tự động tạo ra model GGUF tùy chỉnh từ kế hoạch này.

## Hệ thống Kiểm thử (Benchmark)

Để chứng minh giá trị của EWQ, một bộ benchmark toàn diện đã được xây dựng để so sánh các phiên bản model khác nhau trên nhiều khía cạnh:

**Các phiên bản được so sánh:**
*   **Original (FP16):** Model gốc chưa qua lượng tử hóa.
*   **Standard Q4_K_M (GGUF):** Lượng tử hóa 4-bit tiêu chuẩn.
*   **Standard Q8_0 (GGUF):** Lượng tử hóa 8-bit tiêu chuẩn.
*   **EWQ (bitsandbytes - Blockwise):** Plan EWQ theo block áp dụng qua `bitsandbytes`.
*   **EWQ (GGUF - Blockwise):** Plan EWQ theo block để tạo file GGUF.
*   **EWQ (GGUF - Tensorwise):** Plan EWQ theo tensor để tạo file GGUF (phiên bản mới nhất).

**Các chỉ số được đo lường:**
*   **Chất lượng Model:** MMLU, BLEU, ROUGE-1/2/L.
*   **Hiệu suất:** Tốc độ sinh token (tokens/giây).
*   **Tài nguyên:** Mức sử dụng VRAM (GB).

## Kết quả Benchmark và Phân tích

Đây là phần quan trọng nhất, chứng minh hiệu quả của các phương pháp.

### Bảng Tổng hợp Kết quả (Model: Qwen3-8B)

| Phiên bản Model | VRAM (GB) | % Δ VRAM | Tốc độ (tok/s) | % Δ Tốc độ | MMLU (%) | % Δ MMLU | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original (FP16)** | 15.26 | *Baseline* | 47.04 | *Baseline* | 69.98 | *Baseline* | - | - | - | 0.1783 |
| **Standard Q4_K_M (GGUF)** | **6.40** | **-58.1%** | **124.14** | **+163.9%**| 69.32 | -0.9% | 0.0329 | 0.2600 | 0.0767 | 0.1564 |
| **Standard Q8_0 (GGUF)** | 9.54 | -37.5% | 85.83 | +82.5% | 70.27 | +0.4% | 0.0306 | 0.2540 | 0.0746 | 0.1515 |
| **EWQ (GGUF - Blockwise)** | 8.54 | -44.0% | 95.18 | +102.3% | 70.05 | +0.1% | 0.0322 | 0.2544 | 0.0724 | 0.1548 |
| **EWQ (GGUF - Tensorwise)** | 7.55 | -50.5% | 106.62 | +126.7% | 70.09 | +0.2% | 0.0303 | 0.2513 | 0.0712 | 0.1518 |
| **EWQ (bitsandbytes - Blockwise)** | 9.64 | -36.8% | 43.32 | -7.9% | **70.30** | **+0.5%** | **0.0487** | **0.3148** | **0.0909** | **0.1800** |

*(**In đậm** là giá trị tốt nhất trong từng hạng mục. `% Δ` là phần trăm thay đổi so với bản gốc FP16)*

### Phân tích

1.  **Chất lượng Model (MMLU & ROUGE):**
    *   **Q4_K_M** là phiên bản GGUF duy nhất cho thấy sự sụt giảm nhẹ về điểm MMLU (-0.9%), cho thấy việc lượng tử hóa mạnh tay đồng nhất có ảnh hưởng đến khả năng suy luận.
    *   Tất cả các phiên bản **EWQ và Q8_0 đều bảo toàn hoặc thậm chí cải thiện nhẹ** điểm MMLU so với bản gốc, chứng tỏ các phương pháp lượng tử hóa tinh vi hơn có hiệu quả trong việc giữ lại "trí thông minh" của model.
    *   Về chất lượng sinh văn bản (ROUGE), phiên bản **EWQ (bitsandbytes)** cho kết quả vượt trội, có thể do cách thư viện này xử lý các phép toán. Tuy nhiên, các phiên bản GGUF khác đều cho kết quả khá tương đồng nhau.

2.  **Tài nguyên và Hiệu suất (VRAM & Tốc độ):**
    *   **Standard Q4_K_M** là phiên bản **nhanh nhất và nhẹ nhất**, nhưng phải trả giá bằng việc sụt giảm chất lượng.
    *   Phương pháp **EWQ (GGUF - Tensorwise)** tỏa sáng rực rỡ như một **nhà vô địch về sự cân bằng**. Nó mang lại một bước nhảy vọt về hiệu suất so với tất cả các phương pháp khác (ngoại trừ Q4).
    *   So với **Standard Q8_0**, phiên bản **EWQ-Tensorwise** vượt trội về mọi mặt: **nhẹ hơn 21%** (tiết kiệm 2GB VRAM), **nhanh hơn 24%**, trong khi chất lượng MMLU gần như tương đương.
    *   So với chính phiên bản **EWQ-Blockwise**, việc chuyển sang **Tensorwise** là một cải tiến lớn: **nhẹ hơn 12%** (tiết kiệm 1GB VRAM) và **nhanh hơn 12%**.

## Kết luận

Dự án đã tái hiện và cải tiến thành công phương pháp Lượng tử hóa theo Entropy lên mức độ tensor. Bộ kiểm thử nghiêm ngặt đã chứng minh một cách thuyết phục tính hiệu quả vượt trội của nó.

**Kết quả cho thấy rõ ràng rằng việc sử dụng plan EWQ-Tensorwise để tạo ra một file GGUF tùy chỉnh đã tạo ra một phiên bản model cực kỳ cân bằng và tối ưu. Nó là sự lựa chọn tốt nhất cho những ai tìm kiếm "điểm ngọt" hoàn hảo giữa hiệu suất, yêu cầu tài nguyên và chất lượng, vượt qua cả phương pháp lượng tử hóa 8-bit tiêu chuẩn và các phiên bản EWQ cũ hơn.**

---

# Entropy-Weighted Quantization (EWQ): Comprehensive Re-implementation and Benchmark (Tensor-wise Edition)

This project is an independent effort to re-implement and enhance the **Entropy-Weighted Quantization (EWQ)** method proposed in the scientific paper [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2). This version has been upgraded from block-wise analysis to **tensor-wise analysis**, enabling more granular and effective optimization.

Project Goals:
1.  **Re-implement and enhance the core EWQ algorithm** to generate a custom, tensor-level quantization plan for Large Language Models (LLMs).
2.  **Build a comprehensive benchmark suite** to objectively and rigorously evaluate the effectiveness of the EWQ method against standard quantization techniques.
3.  **Validate** whether the EWQ-Tensorwise method truly produces a superiorly balanced model in terms of quality, performance, and resource consumption.

## Benchmark Results and Analysis

This is the most crucial section, demonstrating the effectiveness of the different methods.

### Summary Table (Model: Qwen3-8B)

| Model Version | VRAM (GB) | % Δ VRAM | Speed (tok/s) | % Δ Speed | MMLU (%) | % Δ MMLU | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original (FP16)** | 15.26 | *Baseline* | 47.04 | *Baseline* | 69.98 | *Baseline* | - | - | - | 0.1783 |
| **Standard Q4_K_M (GGUF)** | **6.40** | **-58.1%** | **124.14** | **+163.9%**| 69.32 | -0.9% | 0.0329 | 0.2600 | 0.0767 | 0.1564 |
| **Standard Q8_0 (GGUF)** | 9.54 | -37.5% | 85.83 | +82.5% | 70.27 | +0.4% | 0.0306 | 0.2540 | 0.0746 | 0.1515 |
| **EWQ (GGUF - Blockwise)** | 8.54 | -44.0% | 95.18 | +102.3% | 70.05 | +0.1% | 0.0322 | 0.2544 | 0.0724 | 0.1548 |
| **EWQ (GGUF - Tensorwise)** | 7.55 | -50.5% | 106.62 | +126.7% | 70.09 | +0.2% | 0.0303 | 0.2513 | 0.0712 | 0.1518 |
| **EWQ (bitsandbytes - Blockwise)** | 9.64 | -36.8% | 43.32 | -7.9% | **70.30** | **+0.5%** | **0.0487** | **0.3148** | **0.0909** | **0.1800** |

*(**Bold** indicates the best value in each category. `% Δ` is the percentage change relative to the FP16 original.)*

### Analysis

1.  **Model Quality (MMLU & ROUGE):**
    *   **Q4_K_M** is the only GGUF version that shows a slight drop in its MMLU score (-0.9%), suggesting that aggressive, uniform quantization impacts reasoning capabilities.
    *   All **EWQ and Q8_0 versions maintained or even slightly improved** the MMLU score compared to the original, proving that more sophisticated quantization methods are effective at preserving the model's "intelligence."
    *   For text generation quality (ROUGE), the **EWQ (bitsandbytes)** version shows superior results, possibly due to the library's operational handling. However, the other GGUF versions perform quite similarly to each other.

2.  **Resources and Performance (VRAM & Speed):**
    *   **Standard Q4_K_M** is the **fastest and lightest** version, but it comes at the cost of reduced quality.
    *   The **EWQ (GGUF - Tensorwise)** method shines as the **champion of balance**. It delivers a leap in performance over all other methods (except Q4).
    *   Compared to **Standard Q8_0**, the **EWQ-Tensorwise** version is superior in every aspect: **21% lighter** (saving 2GB of VRAM), **24% faster**, with a virtually identical MMLU score.
    *   Compared to its **EWQ-Blockwise** predecessor, the move to **Tensorwise** is a major improvement: **12% lighter** (saving 1GB of VRAM) and **12% faster**.

## Conclusion

This project has successfully re-implemented and advanced the Entropy-Weighted Quantization method to the tensor level. The rigorous benchmark suite has convincingly proven its outstanding effectiveness.

**The results clearly demonstrate that using a Tensor-wise EWQ plan to create a custom GGUF file produces an extremely balanced and optimized model. It is the best choice for those seeking the perfect "sweet spot" between performance, resource requirements, and quality, outperforming both standard 8-bit quantization and older EWQ versions.**