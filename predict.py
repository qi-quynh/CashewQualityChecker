import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# ✅ Load model đã huấn luyện
model = tf.keras.models.load_model('cashew_model_3class.h5')

# ✅ Danh sách class theo thứ tự bạn đã huấn luyện
class_names = ['nguyen', 'vo']  # phải đúng thứ tự đã in ra từ train_generator.class_indices

# ✅ Kích thước ảnh đầu vào đúng như lúc huấn luyện
IMG_SIZE = 150

# ✅ Load và xử lý ảnh đầu vào
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa như lúc train
    img_array = np.expand_dims(img_array, axis=0)  # Đưa vào batch có shape (1, 150, 150, 3)

    # ✅ Dự đoán
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Ảnh: {img_path}")
    print(f"✅ Dự đoán: {predicted_class.upper()} ({confidence*100:.2f}%)")

# ✅ Cho phép chạy qua dòng lệnh: python predict.py path/to/image.jpg
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("⚠️  Vui lòng truyền đường dẫn ảnh cần dự đoán.")
    else:
        predict_image(sys.argv[1])
