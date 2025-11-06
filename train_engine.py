# train_engine.py
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from image_processor import extract_features

# --- CÁC THAM SỐ CÓ THỂ ĐIỀU CHỈNH ---
DATA_DIR = 'data_train_augmented'
PROCESSED_DATA_PATH = 'models/processed_data.npz'
ANN_MODEL = 'models/test1.h5'
LABEL_ENCODER_PATH = 'models/class_map.dat'
HISTORY_PLOT_PATH = 'models/training_history_plot.png'
HISTORY_EXCEL_PATH = 'models/training_history.xlsx'
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# Tham số huấn luyện
EPOCHS = 100
BATCH_SIZE = 32
# -----------------------------------------

def process_and_extract_features(data_dir, output_file):
    """Quét qua thư mục dữ liệu, trích xuất đặc trưng từ mỗi ảnh và lưu lại."""
    if os.path.exists(output_file):
        print(f"Tệp {output_file} đã tồn tại. Bỏ qua bước trích xuất đặc trưng.")
        data = np.load(output_file, allow_pickle=True)
        return data['features'], data['labels']

    features_list, labels_list = [], []
    print("\n--- Bắt đầu trích xuất đặc trưng ---")

    if os.path.exists(data_dir):
        for food_folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, food_folder)
            if not os.path.isdir(folder_path): continue
            print(f"Đang xử lý thư mục: {food_folder}")
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                feature_vector = extract_features(img_path)
                if feature_vector is not None:
                    features_list.append(feature_vector)
                    labels_list.append(food_folder)
    
    X, y = np.array(features_list), np.array(labels_list)
    
    if not os.path.exists('models'): os.makedirs('models')
    np.savez_compressed(output_file, features=X, labels=y)
    print(f"✅ Đã xử lý {len(features_list)} ảnh và lưu vào {output_file}")
    return X, y

def build_ANN_model(input_dim, num_classes):
    model = Sequential([
        # --- GỢI Ý ĐIỀU CHỈNH ---
        # 1. Số lượng nơ-ron: Thử thay đổi các giá trị 128, 64. 
        #    Tăng lên (ví dụ: 256, 128) nếu bạn nghĩ mô hình cần phức tạp hơn.
        #    Giảm xuống (ví dụ: 64, 32) nếu mô hình bị overfitting.
        Dense(256, activation='relu', input_dim=input_dim),
        
        # 2. Tỷ lệ Dropout: Đây là tham số chống overfitting rất quan trọng.
        #    Giá trị thường từ 0.2 đến 0.5. Tăng lên nếu overfitting nặng.
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # 3. Tốc độ học (Learning Rate):
    #    Giá trị mặc định của Adam là 0.001. Thử các giá trị nhỏ hơn
    #    (ví dụ: 0.0005) nếu loss không giảm đều, hoặc lớn hơn một chút.
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = process_and_extract_features(DATA_DIR, PROCESSED_DATA_PATH)
    
    if X.size == 0 or y.size == 0:
        print("\nLỖI: Không có dữ liệu để huấn luyện. Dừng chương trình.")
    else:
        # Chuẩn bị dữ liệu
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_categorical
        )
        
        print("\n--- Bắt đầu huấn luyện mô hình ")
        
        # Xây dựng và huấn luyện
        model = build_ANN_model(X_train.shape[1], num_classes)
        
        if os.path.exists(ANN_MODEL): os.remove(ANN_MODEL)
        if os.path.exists(LABEL_ENCODER_PATH): os.remove(LABEL_ENCODER_PATH)
        
        checkpoint = ModelCheckpoint(ANN_MODEL, monitor='val_accuracy', save_best_only=True, mode='max')
        # --- GỢI Ý ĐIỀU CHỈNH ---
        # 4. Patience trong EarlyStopping: Số epoch chờ đợi trước khi dừng nếu không có cải thiện.
        #    Tăng giá trị này (ví dụ: 20-25) nếu bạn muốn mô hình kiên nhẫn học lâu hơn.
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min')

        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                            validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping], verbose=1)
        
        # Lưu kết quả
        joblib.dump(le, LABEL_ENCODER_PATH)
        
        history_df = pd.DataFrame(history.history)
        history_df.to_excel(HISTORY_EXCEL_PATH, index=False)
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc'); plt.legend(); plt.title('Accuracy')
        plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.legend(); plt.title('Loss')
        plt.savefig(HISTORY_PLOT_PATH)
        plt.close() # Thêm dòng này để đóng cửa sổ plot sau khi lưu
        print(f"✅ Đã lưu kết quả huấn luyện (Excel, Biểu đồ) vào '{HISTORY_EXCEL_PATH}' và '{HISTORY_PLOT_PATH}'.")