import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
# Các lớp cốt lõi của CNN: Tích chập, Gộp, Làm phẳng, và Kết nối đầy đủ
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical, load_img, img_to_array 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
# Không còn cần import extract_features

# --- CÁC THAM SỐ CÓ THỂ ĐIỀU CHỈNH ---
# LƯU Ý: Thay đổi đường dẫn này trên máy tính của bạn (D:\phanloaidoan\data_train_augmented)
DATA_DIR = 'data_train_augmented' 
PROCESSED_DATA_PATH = 'models/cnn_processed_data.npz'
CNN_MODEL_PATH = 'models/test2.h5'
LABEL_ENCODER_PATH = 'models/cnn_class_map.dat'
HISTORY_PLOT_PATH = 'models/cnn_training_history_plot.png'
HISTORY_EXCEL_PATH = 'models/cnn_training_history.xlsx'

# Tham số xử lý ảnh
IMG_SIZE = (128, 128) # Kích thước chuẩn hóa ảnh
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# Tham số huấn luyện
EPOCHS = 50 
BATCH_SIZE = 32
# -----------------------------------------

def process_and_load_images(data_dir, output_file, img_size):
    """Quét qua thư mục dữ liệu, tải ảnh, chuẩn hóa và lưu lại dưới dạng Tensor."""
    if os.path.exists(output_file):
        print(f"Tệp {output_file} đã tồn tại. Bỏ qua bước tải ảnh.")
        data = np.load(output_file, allow_pickle=True)
        return data['images'], data['labels']

    images_list, labels_list = [], []
    print("\n--- Bắt đầu tải và chuẩn hóa ảnh thành Tensor ---")

    if os.path.exists(data_dir):
        for food_folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, food_folder)
            if not os.path.isdir(folder_path): continue
            print(f"Đang xử lý thư mục: {food_folder}")
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    # Tải ảnh, Resize, và chuyển thành mảng numpy (tensor 3 chiều)
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    
                    # Chuẩn hóa pixel về [0, 1]
                    img_array = img_array / 255.0
                    
                    images_list.append(img_array)
                    labels_list.append(food_folder)
                except Exception:
                    # Bỏ qua các file không hợp lệ
                    pass
                    
    X, y = np.array(images_list), np.array(labels_list)
    
    if not os.path.exists('models'): os.makedirs('models')
    np.savez_compressed(output_file, images=X, labels=y)
    print(f"✅ Đã xử lý {len(images_list)} ảnh và lưu vào {output_file}")
    return X, y

def build_CNN_model(input_shape, num_classes):
    """Xây dựng kiến trúc mô hình Mạng nơ-ron Tích chập (CNN)."""
    model = Sequential([
        # PHẦN HỌC ĐẶC TRƯNG TỰ ĐỘNG
        
        # Tầng Tích chập (Học các đặc trưng cơ bản)
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)), # Giảm kích thước
        
        # Tầng Tích chập thứ hai (Học các đặc trưng phức tạp)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Tầng Tích chập thứ ba (Học các đặc trưng cấp cao)
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # PHẦN PHÂN LOẠI
        
        # Làm phẳng Tensor đặc trưng
        Flatten(),
        
        # Tầng Kết nối Đầy đủ (Phân loại)
        Dense(512, activation='relu'),
        Dropout(0.5), # Kỹ thuật chống quá khớp
        
        # Tầng đầu ra
        Dense(num_classes, activation='softmax')
    ])
    
    # Cấu hình tối ưu hóa và hàm mất mát
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    # Bước 1: Tải và chuẩn hóa ảnh
    X, y = process_and_load_images(DATA_DIR, PROCESSED_DATA_PATH, IMG_SIZE)
    
    if X.size == 0 or y.size == 0:
        print("\nLỖI: Không có dữ liệu để huấn luyện. Dừng chương trình.")
    else:
        # Chuẩn bị dữ liệu
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        y_categorical = to_categorical(y_encoded, num_classes=num_classes)
        
        # Chia tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y_categorical
        )
        
        # Xác định hình dạng đầu vào cho CNN
        input_shape = X_train.shape[1:]
        
        print("\n--- Bắt đầu huấn luyện mô hình CNN ---")
        
        # Xây dựng và huấn luyện
        model = build_CNN_model(input_shape, num_classes)
        
        # Xóa các tệp cũ để lưu mô hình và encoder mới
        if os.path.exists(CNN_MODEL_PATH): os.remove(CNN_MODEL_PATH)
        if os.path.exists(LABEL_ENCODER_PATH): os.remove(LABEL_ENCODER_PATH)
        
        checkpoint = ModelCheckpoint(CNN_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min') 

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
        plt.close()
        print(f"✅ Đã lưu kết quả huấn luyện CNN (Excel, Biểu đồ) vào '{HISTORY_EXCEL_PATH}' và '{HISTORY_PLOT_PATH}'.")