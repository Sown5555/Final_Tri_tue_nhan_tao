# test_grid_crop.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CẤU HÌNH ---
IMAGE_PATH = 'data_test/khay_day_du/alo.jpg' 
PADDING_PERCENT = 0.1
# -----------------

def preprocess_image_to_4_3(image):
    h, w, _ = image.shape
    rotated_image = image.copy()
    if h > w:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w, _ = rotated_image.shape
    
    target_aspect = 4.0 / 3.0
    current_aspect = w / float(h)
    
    if current_aspect > target_aspect:
        new_w = int(target_aspect * h)
        x_start = (w - new_w) // 2
        cropped_4_3 = rotated_image[:, x_start:x_start + new_w]
    elif current_aspect < target_aspect:
        new_h = int(w / target_aspect)
        y_start = (h - new_h) // 2
        cropped_4_3 = rotated_image[y_start:y_start + new_h, :]
    else:
        cropped_4_3 = rotated_image

    return cropped_4_3, rotated_image

def apply_padding(rect, padding_percent):
    x, y, w, h = rect
    pad_x = int(w * padding_percent); pad_y = int(h * padding_percent)
    return (x + pad_x, y + pad_y, w - (2 * pad_x), h - (2 * pad_y))

if __name__ == "__main__":
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None: print(f"Lỗi: Không tìm thấy ảnh tại '{IMAGE_PATH}'"); exit()

    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    final_4_3_image, rotated_image = preprocess_image_to_4_3(original_image_rgb)
    h, w, _ = final_4_3_image.shape

    # === THAY ĐỔI TỈ LỆ NGANG Ở ĐÂY ===
    split_y = int(h * 0.55) # 55% trên, 45% dưới
    # =================================

    top_left_w = int(w * 0.6)
    top_left_rect = (0, 0, top_left_w, split_y)
    top_right_rect = (top_left_w, 0, w - top_left_w, split_y)
    bottom_h = h - split_y; bottom_w = w // 3
    bottom_left_rect = (0, split_y, bottom_w, bottom_h)
    bottom_center_rect = (bottom_w, split_y, bottom_w, bottom_h)
    bottom_right_rect = (bottom_w * 2, split_y, w - (bottom_w * 2), bottom_h)
    all_rects = [top_left_rect, top_right_rect, bottom_left_rect, bottom_center_rect, bottom_right_rect]
    
    cropped_regions = [final_4_3_image[py:py+ph, px:px+pw] for (px, py, pw, ph) in [apply_padding(r, PADDING_PERCENT) for r in all_rects]]

    # (Phần code hiển thị plt giữ nguyên)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Quy trình Xử lý và Cắt ảnh (Tỉ lệ 55:45)", fontsize=16)
    ax_flat = axes.flatten()
    for i in range(len(cropped_regions) + 3, len(ax_flat)): ax_flat[i].axis('off')
    ax_flat[0].imshow(original_image_rgb); ax_flat[0].set_title("1. Ảnh Gốc"); ax_flat[0].axis('off')
    ax_flat[1].imshow(rotated_image); ax_flat[1].set_title("2. Ảnh đã Xoay"); ax_flat[1].axis('off')
    display_grid = final_4_3_image.copy()
    for (x, y, w, h) in all_rects: cv2.rectangle(display_grid, (x, y), (x + w, y + h), (0, 255, 0), 5)
    ax_flat[2].imshow(display_grid); ax_flat[2].set_title("3. Ảnh 4:3 và Lưới chia"); ax_flat[2].axis('off')
    titles = ["4. Top Left", "5. Top Right", "6. Bottom Left", "7. Bottom Center", "8. Bottom Right"]
    for i, img in enumerate(cropped_regions):
        ax_flat[i+3].imshow(img); ax_flat[i+3].set_title(titles[i]); ax_flat[i+3].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()