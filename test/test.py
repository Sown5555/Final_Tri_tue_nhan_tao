import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Import các module xử lý bạn đã tạo
# Đảm bảo các file này nằm cùng thư mục hoặc có thể được Python tìm thấy
import tes1
import image_processor

def cut_and_predict_tray(image_path: str):
    """
    Hàm chính để tải ảnh, cắt thành 5 phần, dự đoán và hiển thị kết quả.

    Args:
        image_path (str): Đường dẫn đến tệp ảnh khay cơm.
    """
    # --- Bước 1: Kiểm tra và tải ảnh ---
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy ảnh tại đường dẫn: '{image_path}'")
        return

    # cv2.imread đọc ảnh theo định dạng BGR (Blue-Green-Red)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Lỗi: Không thể đọc được file ảnh. File có thể bị hỏng hoặc không phải định dạng ảnh.")
        return
        
    print(f"✅ Đã tải ảnh '{os.path.basename(image_path)}' thành công.")

    # --- Bước 2: Cắt ảnh theo logic của ứng dụng ---
    print("...Đang cắt ảnh thành các phần...")
    
    # Sử dụng hàm crop_to_4_3 từ image_processor của bạn
    final_img_bgr, _ = image_processor.crop_to_4_3(image_bgr)
    h, w, _ = final_img_bgr.shape

    # Tọa độ và kích thước của 5 vùng cắt (rectangles)
    # (x_start, y_start, width, height)
    rects = [
        (0, 0, int(w * 0.6), int(h * 0.55)),         # Vùng 1: Món chính
        (int(w * 0.6), 0, int(w * 0.4), int(h * 0.55)), # Vùng 2: Món phụ trên
        (0, int(h * 0.55), w // 3, int(h * 0.45)),     # Vùng 3: Món phụ dưới trái
        (w // 3, int(h * 0.55), w // 3, int(h * 0.45)),  # Vùng 4: Món phụ dưới giữa
        (w // 3 * 2, int(h * 0.55), w - (w // 3 * 2), int(h * 0.45)) # Vùng 5: Món phụ dưới phải
    ]

    results = [] # List để lưu ảnh con và kết quả dự đoán

    # --- Bước 3: Dự đoán từng phần ảnh ---
    print("...Đang dự đoán từng món ăn bằng AI...")
    for i, (x, y, w_r, h_r) in enumerate(rects):
        # Cắt ảnh món ăn từ ảnh gốc đã được crop
        item_img_bgr = final_img_bgr[y:y + h_r, x:x + w_r]

        prediction_text = "Không nhận dạng được"
        if item_img_bgr.size > 0:
            # Gọi hàm execute từ tes1.py để dự đoán
            pred_result = tes1.execute(item_img_bgr)
            prediction_text = pred_result.get("final_prediction", "Lỗi dự đoán")
        
        # Chuyển đổi màu từ BGR (OpenCV) sang RGB (Matplotlib) để hiển thị đúng
        item_img_rgb = cv2.cvtColor(item_img_bgr, cv2.COLOR_BGR2RGB)
        
        results.append({
            "image": item_img_rgb,
            "prediction": f"Món #{i+1}: {prediction_text}"
        })

    print("✅ Hoàn tất dự đoán!")

    # --- Bước 4: Hiển thị kết quả bằng Matplotlib ---
    print("...Đang hiển thị kết quả...")
    # Tạo một cửa sổ hình ảnh với 2 hàng, 3 cột
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Kết quả phân tích khay cơm", fontsize=20, fontweight='bold')

    # `axes.flatten()` biến lưới 2x3 thành một mảng phẳng để dễ lặp
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.imshow(res["image"])
        ax.set_title(res["prediction"], fontsize=12)
        ax.axis('off') # Ẩn các trục tọa độ (x, y)

    # Ẩn ô subplot cuối cùng vì chỉ có 5 ảnh
    axes[5].axis('off')
            
    # Tự động điều chỉnh layout để không bị chồng chéo
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Chừa không gian cho suptitle
    plt.show()


# --- PHẦN THỰC THI CHÍNH ---
if __name__ == "__main__":
    # ⚠️ THAY ĐỔI ĐƯỜNG DẪN Ở ĐÂY
    # Sử dụng r"..." để Windows xử lý dấu gạch chéo ngược `\` một cách chính xác
    path_to_your_image = r"F:\phanloaidoan\data_test\khay_day_du\alo5.jpg"
    
    # Gọi hàm chính để bắt đầu
    cut_and_predict_tray(path_to_your_image)