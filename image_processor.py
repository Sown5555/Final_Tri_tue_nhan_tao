# image_processor.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# --- CÁC THAM SỐ CÓ THỂ ĐIỀU CHỈNH ---
# Bạn có thể thử nghiệm với các kích thước khác nhau để xem hiệu quả.
RESIZE_DIM = (200, 200) 
# Số lượng điểm lân cận và bán kính cho thuật toán LBP (trích xuất kết cấu).
LBP_POINTS = 24
LBP_RADIUS = 3
# Số lượng bin cho biểu đồ màu.
HIST_BINS = 16
# -----------------------------------------

def crop_to_4_3(image):
    h, w = image.shape[:2]
    
    rotated_image = image.copy()
    if h > w:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = rotated_image.shape[:2]
    
    target_aspect = 4.0 / 3.0
    current_aspect = w / float(h)
    
    if current_aspect > target_aspect:
        new_w = int(target_aspect * h)
        x_start = (w - new_w) // 2
        cropped_image = rotated_image[:, x_start:x_start + new_w]
    else:
        new_h = int(w / target_aspect)
        y_start = (h - new_h) // 2
        cropped_image = rotated_image[y_start:y_start + new_h, :]

    return cropped_image, rotated_image

def extract_features(image_source):
    """
    Trích xuất đặc trưng từ ảnh.
    Hàm này có thể nhận đầu vào là đường dẫn file (str) hoặc dữ liệu ảnh (np.ndarray).
    """
    try:
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None: return None
        elif isinstance(image_source, np.ndarray):
            image = image_source
        else:
            return None # Không hỗ trợ định dạng này

        if image is None or image.size == 0: return None
        
        image = cv2.resize(image, RESIZE_DIM)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist_hue = cv2.calcHist([hsv], [0], None, [HIST_BINS], [0, 180])
        cv2.normalize(hist_hue, hist_hue)
        hist_sat = cv2.calcHist([hsv], [1], None, [HIST_BINS], [0, 256])
        cv2.normalize(hist_sat, hist_sat)
        color_features = np.concatenate([hist_hue, hist_sat]).flatten()

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_features = np.zeros(7)
        if contours:
            moments = cv2.moments(max(contours, key=cv2.contourArea))
            shape_features = cv2.HuMoments(moments).flatten()

        lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
        (texture_features, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
        texture_features = texture_features.astype("float") / (texture_features.sum() + 1e-6)

        return np.hstack([color_features, shape_features, texture_features])
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng: {e}")
        return None