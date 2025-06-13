# EWQ: Lượng Tử Hóa Trọng Số Dựa Trên Entropy - Tăng Tốc LLM và Giảm VRAM

Đây là mã nguồn Proof-of-Concept (PoC) cho phương pháp **Lượng tử hóa Trọng số dựa trên Entropy (Entropy-based Weight Quantization - EWQ)**, một kỹ thuật nhằm tối ưu hóa các Mô hình Ngôn ngữ Lớn (LLM).

## 🚀 Giới thiệu

Khi đọc bài báo [arXiv:2503.04704v2](https://arxiv.org/html/2503.04704v2), tôi nhận thấy các tác giả đã đề xuất một hướng đi thú vị nhưng không cung cấp mã nguồn để kiểm chứng. Vì vậy, tôi đã tự mình xây dựng một hệ thống để triển khai và đánh giá ý tưởng cốt lõi: **Không phải tất cả các layer trong một LLM đều quan trọng như nhau, và chúng ta có thể lượng tử hóa chúng một cách có chọn lọc.**

Dự án này ra đời để chứng minh rằng bằng cách phân tích độ phức tạp (entropy) của từng khối trong mô hình, chúng ta có thể tạo ra một "kế hoạch lượng tử hóa" thông minh, giúp giảm đáng kể mức sử dụng VRAM và tăng tốc độ xử lý mà không cần thay đổi kiến trúc.

## 💡 Phương pháp thực hiện (The EWQ Method)

Ý tưởng chính đằng sau EWQ rất đơn giản:

1.  **Giả thuyết:** Các khối (layers) trong LLM có mức độ nhạy cảm khác nhau đối với việc lượng tử hóa. Các khối có trọng số phức tạp hơn (entropy cao) nên được giữ ở độ chính xác cao, trong khi các khối đơn giản hơn (entropy thấp) có thể bị lượng tử hóa mạnh hơn.
2.  **Phân tích Entropy:** Hệ thống sẽ tải mô hình gốc lên CPU và tính toán entropy trung bình của trọng số cho từng khối transformer.
3.  **Tạo Kế hoạch:** Dựa trên phân phối entropy của tất cả các khối, một "kế hoạch lượng tử hóa" (`quant_plan.json`) được tạo ra. Kế hoạch này chỉ định độ chính xác cho từng khối:
    *   **`raw` (FP16):** Dành cho các khối có entropy cao nhất (nhạy cảm nhất).
    *   **`8-bit`:** Dành cho các khối có entropy trung bình.
    *   **`4-bit`:** Dành cho các khối có entropy thấp nhất (ít nhạy cảm nhất).
4.  **Áp dụng và Tối ưu:** Mô hình được lượng tử hóa theo kế hoạch đã tạo, sau đó được chuyển sang GPU để chạy benchmark. Toàn bộ quá trình này được thiết kế để tối ưu hóa việc sử dụng VRAM.

## 📊 Kết quả đạt được

Chúng tôi đã thực hiện benchmark trên model `unsloth/Meta-Llama-3.1-8B-Instruct` và kết quả thật sự ấn tượng.

| Chỉ số | Mô hình Gốc (FP16) | Mô hình Lượng tử hóa EWQ | Thay đổi |
| :--- | :---: | :---: | :---: |
| **Sử dụng VRAM** | ~14.97 GB | **~11.43 GB** | **Giảm 24%** (Tiết kiệm 3.54 GB) |
| **Tốc độ trung bình** | ~50.79 tokens/s | **~57.79 tokens/s** | **Nhanh hơn 14%** |

Phương pháp EWQ đã tạo ra một mô hình **vừa nhanh hơn, vừa nhẹ hơn** một cách đáng kể. Đây là một kết quả "win-win", cho thấy tiềm năng to lớn của việc lượng tử hóa có chọn lọc.

## ⚙️ Quy trình hoạt động

Hệ thống được chia thành 3 kịch bản chính để đảm bảo tính module và hiệu quả:

1.  **`main_cache_model.py` (Tạo kế hoạch):**
    *   Tải model gốc lên **CPU** (để không tốn VRAM).
    *   Phân tích entropy của từng khối.
    *   Tạo và lưu file `quant_plan_{model_hash}.json` trong thư mục `quantized_models`.

2.  **`benchmark.py` (Benchmark mô hình EWQ):**
    *   Tải kế hoạch lượng tử hóa đã được tạo trước đó.
    *   Tải model gốc lên CPU, áp dụng kế hoạch lượng tử hóa (kết hợp các layer FP16, 8-bit, 4-bit).
    *   Chuyển model đã được lượng tử hóa hoàn chỉnh sang GPU.
    *   Chạy bộ benchmark toàn diện và lưu kết quả vào thư mục `benchmark_results`.

3.  **`benchmark_original.py` (Benchmark mô hình gốc):**
    *   Tải thẳng model gốc lên GPU và chạy cùng bộ benchmark để có một đường cơ sở (baseline) so sánh.

## 🚀 Hướng dẫn sử dụng

Để tái tạo lại kết quả này:

1.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Bước 1: Tạo kế hoạch lượng tử hóa (chạy trên CPU)**
    Chạy script này để phân tích mô hình và tạo file kế hoạch.
    ```bash
    python main_cache_model.py
    ```

3.  **Bước 2: Chạy benchmark cho mô hình đã tối ưu bằng EWQ**
    Sau khi kế hoạch đã được tạo, chạy script này để lượng tử hóa và đo lường hiệu năng.
    ```bash
    python benchmark.py
    ```

4.  **(Tùy chọn) Bước 3: Chạy benchmark cho mô hình gốc để so sánh**
    ```bash
    python benchmark_original.py
    ```

5.  **Kiểm tra kết quả:**
    Tất cả các file JSON chứa kết quả chi tiết sẽ được lưu trong thư mục `benchmark_results`.

## 📁 Cấu trúc thư mục
    .
    ├── benchmark.py # Script benchmark mô hình EWQ
    ├── benchmark_original.py # Script benchmark mô hình gốc
    ├── main_cache_model.py # Script tạo kế hoạch lượng tử hóa
    │
    ├── models/ # Thư mục cache cho model gốc từ Hugging Face
    ├── quantized_models/ # Thư mục chứa các file kế hoạch lượng tử hóa
    │ └── quant_plan_...json
    │
    └── benchmark_results/ # Thư mục chứa kết quả benchmark
    ├── ewq_benchmark_...json
    └── original_benchmark_...json


## 🔮 Hướng phát triển trong tương lai

*   Đánh giá chất lượng đầu ra của mô hình (เช่น bằng BLEU, ROUGE) để đảm bảo không bị suy giảm hiệu suất.
*   Thử nghiệm với các mô hình và kiến trúc khác nhau.
*   Tinh chỉnh thuật toán xác định ngưỡng entropy để tối ưu hơn nữa.

Cảm ơn bạn đã xem qua dự án này. Mọi đóng góp và ý kiến đều được chào đón!