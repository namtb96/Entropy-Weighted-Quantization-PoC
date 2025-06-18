# =================================================================
# == ĐOẠN MÃ CHẨN ĐOÁN - ĐẶT NGAY TRÊN ĐẦU FILE CỦA BẠN =========
# =================================================================
import llama_cpp
import sys

print("\n\n--- THÔNG TIN CHẨN ĐOÁN MÔI TRƯỜNG ---")
try:
    print(f"Phiên bản llama-cpp-python ĐANG CHẠY: {llama_cpp.__version__}")
except AttributeError:
    print("Phiên bản llama-cpp-python quá cũ, không có thuộc tính __version__.")

print(f"Đường dẫn thực thi Python: {sys.executable}")
print(f"Vị trí của module llama_cpp: {llama_cpp.__file__}")
print("-------------------------------------------\n\n")
# =================================================================
# == KẾT THÚC ĐOẠN MÃ CHẨN ĐOÁN ===================================
# =================================================================


# Code còn lại của bạn bắt đầu từ đây
# Ví dụ: import torch, import numpy, v.v.
# ...