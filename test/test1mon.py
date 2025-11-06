import cv2
import matplotlib.pyplot as plt
import os
import tes1

def du_doan_mot_mon(image_path: str):
    """
    Hàm này tải một ảnh chứa một món ăn, dự đoán và hiển thị kết quả.
    """
    # --- Bước 1: Kiểm tra và tải ảnh ---
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy ảnh tại đường dẫn: '{image_path}'")
        return

    # Sử dụng OpenCV để đọc ảnh, đảm bảo đồng nhất với ứng dụng
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Lỗi: Không thể đọc được file ảnh. File có thể bị hỏng.")
        return
        
    print(f"✅ Đã tải ảnh '{os.path.basename(image_path)}' thành công.")
 
    # Gọi thẳng hàm execute với toàn bộ ảnh
    prediction_result = tes1.execute(image_bgr)
    
    # Lấy ra tên món ăn dự đoán được
    final_prediction = prediction_result.get("final_prediction", "Không nhận dạng được")
    
    print(f"✅ Kết quả dự đoán: {final_prediction}")

    # --- Bước 3: Hiển thị ảnh và kết quả ---
    print("...Đang hiển thị kết quả...")

    # Chuyển đổi màu từ BGR (của OpenCV) sang RGB (của Matplotlib)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Tạo cửa sổ hiển thị
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(f"AI dự đoán là: {final_prediction}", fontsize=14)
    plt.axis('off') # Ẩn các trục tọa độ
    plt.show()

# --- PHẦN THỰC THI CHÍNH ---
if __name__ == "__main__":
    # ⚠️ THAY ĐỔI ĐƯỜNG DẪN Ở ĐÂY
    # Hãy trỏ đến file ảnh chứa một món ăn bạn muốn kiểm tra
    path_to_your_image = r"F:\phanloaidoan\data_test\mon_le\alo.jpg"
    
    # Gọi hàm chính để bắt đầu
    du_doan_mot_mon(path_to_your_image)