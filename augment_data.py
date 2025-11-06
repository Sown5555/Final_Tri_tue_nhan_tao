import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# --- PHẦN CẤU HÌNH ---
SOURCE_DIR = 'data_train'           # Thư mục chứa ảnh gốc
TARGET_DIR = 'data_train_augmented' # Thư mục để lưu ảnh mới
AUGMENTATIONS_PER_IMAGE = 10        # Số lượng ảnh mới muốn tạo ra từ mỗi ảnh gốc

# --- THIẾT LẬP CÁC PHÉP BIẾN ĐỔI ---
datagen = ImageDataGenerator(
    rotation_range=40,          # <-- Tăng độ xoay
    width_shift_range=0.25,     # <-- Tăng dịch chuyển ngang
    height_shift_range=0.25,    # <-- Tăng dịch chuyển dọc
    shear_range=0.2,
    zoom_range=0.25,            # <-- Tăng độ phóng to/thu nhỏ
    horizontal_flip=True,
    brightness_range=[0.7, 1.3], # <-- Tăng dải độ sáng
    fill_mode='nearest'
)

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Xóa thư mục đích nếu nó đã tồn tại để tránh trùng lặp
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    
    os.makedirs(TARGET_DIR)
    
    print("Bắt đầu quá trình tăng cường dữ liệu...")
    
    # Lặp qua các thư mục món ăn trong thư mục gốc
    for food_folder in os.listdir(SOURCE_DIR):
        source_folder_path = os.path.join(SOURCE_DIR, food_folder)
        target_folder_path = os.path.join(TARGET_DIR, food_folder)

        if not os.path.isdir(source_folder_path):
            continue

        os.makedirs(target_folder_path)
        print(f"Đang xử lý thư mục: {food_folder}")

        # Lặp qua từng ảnh trong thư mục món ăn
        for img_name in os.listdir(source_folder_path):
            img_path = os.path.join(source_folder_path, img_name)

            # 1. Sao chép ảnh gốc vào thư mục mới
            shutil.copy(img_path, target_folder_path)
            
            # 2. Tải ảnh và chuẩn bị để tăng cường
            image = cv2.imread(img_path)
            # Chuyển đổi ảnh thành một "batch" có 1 ảnh
            image = np.expand_dims(image, axis=0)

            # 3. Tạo ra các phiên bản ảnh mới
            i = 0
            # datagen.flow() sẽ tạo ra các ảnh biến đổi một cách vô hạn
            for batch in datagen.flow(image, batch_size=1):
                # Lấy ảnh đã biến đổi ra khỏi batch
                augmented_image = batch[0].astype('uint8')
                
                # Tạo tên file mới
                base_name, extension = os.path.splitext(img_name)
                new_img_name = f"{base_name}_aug_{i}{extension}"
                new_img_path = os.path.join(target_folder_path, new_img_name)
                
                # Lưu ảnh mới
                cv2.imwrite(new_img_path, augmented_image)
                
                i += 1
                if i >= AUGMENTATIONS_PER_IMAGE:
                    break  # Dừng lại khi đã tạo đủ số lượng ảnh

    print("\n--- HOÀN TẤT TĂNG CƯỜNG DỮ LIỆU! ---")
    print(f"Dữ liệu mới đã được lưu tại thư mục: {TARGET_DIR}")