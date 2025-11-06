import os
import joblib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
from image_processor import extract_features

# --- CÁC ĐƯỜNG DẪN MÀ BẠN ĐÃ ĐẶT ---
test1 = "models/test1.h5" 
test2 = "models/test2.h5"
monchay = "models/classchay_map.dat"
monman = "models/classman_map.dat"
LABEL_PATH = "models/class_map.dat"

# Đồng bộ kích thước ảnh với lúc huấn luyện
CNN_IMG_SIZE = (192, 192) 

print("Core Utils: Đang tải mô hình CNN...")

# =========================
# 1. Tải Label Encoder
# =========================
_le = None
_num_classes = 0
try:
    _le = joblib.load(LABEL_PATH)
    _num_classes = len(_le.classes_)
    print(f"✅ Đã tải Label Encoder với {_num_classes} lớp.")
except Exception as e:
    print(f"❌ Lỗi: Không thể tải file class_map.dat. Lỗi: {e}")

# =========================
# 2. Tải mô hình
# =========================
def load_cnn_model():
    if not os.path.exists(test1):
        print(f"⚠️ Không tìm thấy model tại {test1}")
        return None
    try:
        model = load_model(test1, compile=False)
        print("✅ Đã tải mô hình thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None
_core_cnn = load_cnn_model()
def load_generic_model(path, model_type='joblib'):
    if path is None or not os.path.exists(path):
        return None
    try:
        if model_type == 'keras':
            model = load_model(path, compile=False)
        else:
            model = joblib.load(path)
        return model
    except Exception as e:
        return None
_core_ = load_generic_model(test2, model_type='keras')
_comp_a = load_generic_model(monchay)
_comp_b = load_generic_model(monman)
print("✅ Hoàn tất quá trình khởi tạo mô hình!")
def _get_top_n(probabilities, n=3):
    if _le is None or not isinstance(probabilities, np.ndarray) or probabilities.size == 0:
        return []
    top_indices = probabilities.argsort()[-n:][::-1]
    valid_indices = [i for i in top_indices if i < len(_le.classes_)]
    if not valid_indices: return []
    labels = _le.inverse_transform(valid_indices)
    probs = probabilities[valid_indices]
    return list(zip(labels, probs))
def execute(image_data):
    if _le is None: return {"error": "Label Encoder chưa được tải."}
    if _core_cnn is None: return {"error": "Mô hình chưa được tải."}
    feature_vector_2d = None
    if any(model is not None for model in [_core_, _comp_a, _comp_b]):
        feature_vector = extract_features(image_data)
        if feature_vector is not None:
            feature_vector_2d = np.array(feature_vector).reshape(1, -1)
    img_resized = cv2.resize(image_data, CNN_IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)
    p_cnn = _core_cnn.predict(img_batch, verbose=0)[0]
    p_cnn1 = _core_.predict(feature_vector_2d, verbose=0)[0] if _core_ and feature_vector_2d is not None else np.zeros(_num_classes)
    p_cnn2 = _comp_a.predict_proba(feature_vector_2d)[0] if _comp_a and feature_vector_2d is not None else np.zeros(_num_classes)
    p_cnn3 = _comp_b.predict_proba(feature_vector_2d)[0] if _comp_b and feature_vector_2d is not None else np.zeros(_num_classes)
    top3_cnn = _get_top_n(p_cnn)
    top3_cnn1 = _get_top_n(p_cnn1)
    top3_cnn2 = _get_top_n(p_cnn2)
    top3_cnn3 = _get_top_n(p_cnn3)
    final_prediction = None
    decision_method = ""
    def find_consensus(predictions_with_source):
        vote_counts = Counter(p[1] for p in predictions_with_source)
        if not vote_counts:
            return None, None
        most_common_pred, count = vote_counts.most_common(1)[0]
        if count >= 3:
            return most_common_pred, f"Đồng thuận ({count}/{len(predictions_with_source)} models)"
        if count == 2:
            models_that_voted = {p[0] for p in predictions_with_source if p[1] == most_common_pred}
            if models_that_voted == {'mlp', 'rf'}:
                return most_common_pred, "Đồng thuận đặc biệt "
        return None, None
    top1_with_source = []
    if top3_cnn: top1_with_source.append(('cnn', top3_cnn[0][0]))
    if top3_cnn1: top1_with_source.append(('cnn1', top3_cnn1[0][0]))
    if top3_cnn2: top1_with_source.append(('cnn2', top3_cnn2[0][0]))
    if top3_cnn3: top1_with_source.append(('cnn3', top3_cnn3[0][0]))
    final_prediction, method = find_consensus(top1_with_source)
    if final_prediction:
        decision_method = f"{method} ở Top 1"
    if not final_prediction:
        top2_with_source = []
        if len(top3_cnn) > 1: top2_with_source.append(('cnn', top3_cnn[1][0]))
        if len(top3_cnn1) > 1: top2_with_source.append(('cnn1', top3_cnn1[1][0]))
        if len(top3_cnn2) > 1: top2_with_source.append(('cnn2', top3_cnn2[1][0]))
        if len(top3_cnn3) > 1: top2_with_source.append(('cnn3', top3_cnn3[1][0]))
        final_prediction, method = find_consensus(top2_with_source)
        if final_prediction:
            decision_method = f"{method} ở Top 2"
    if not final_prediction:
        top3_with_source = []
        if len(top3_cnn) > 2: top3_with_source.append(('cnn', top3_cnn[2][0]))
        if len(top3_cnn1) > 2: top3_with_source.append(('cnn1', top3_cnn1[2][0]))
        if len(top3_cnn2) > 2: top3_with_source.append(('cnn2', top3_cnn2[2][0]))
        if len(top3_cnn3) > 2: top3_with_source.append(('cnn3', top3_cnn3[2][0]))
        final_prediction, method = find_consensus(top3_with_source)
        if final_prediction:
            decision_method = f"{method} ở Top 3"
    if not final_prediction:
        all_preds = [p[0] for top3_list in [top3_cnn, top3_cnn1, top3_cnn2, top3_cnn3] if top3_list for p in top3_list]
        if not all_preds:
            return {"error": "Không có dự đoán nào được tạo ra."}
        vote_counts = Counter(all_preds)
        max_votes = vote_counts.most_common(1)[0][1]
        top_candidates = [item for item, count in vote_counts.items() if count == max_votes]
        if len(top_candidates) == 1:
            final_prediction = top_candidates[0]
            decision_method = "Số đông trên toàn bộ Top 3"
        else:
            tie_breaker_scores = {}
            for top3_list in [top3_cnn, top3_cnn1, top3_cnn2, top3_cnn3]:
                if top3_list:
                    for label, prob in top3_list:
                        if label in top_candidates:
                            tie_breaker_scores[label] = tie_breaker_scores.get(label, 0) + prob
            if tie_breaker_scores:
                final_prediction = max(tie_breaker_scores, key=tie_breaker_scores.get)
                decision_method = "Số đông"
    return {
        "final_prediction": final_prediction,
        "decision_method": decision_method,
        "cnn_top_3": top3_cnn,
        "cnn1_top_3": top3_cnn1,
        "cnn2_top_3": top3_cnn2,
        "cnn3_top_3": top3_cnn3
    }