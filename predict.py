import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load model đã huấn luyện
model = tf.keras.models.load_model('cashew_model_3class.h5')

# Lấy class names đúng theo thứ tự index từ train_generator.class_indices
# Ví dụ: {'con_vo': 0, 'nguyen': 1, 'vo': 2}
class_names = ['con_vo', 'nguyen', 'vo']

# Kích thước ảnh đầu vào (phải đúng như khi train)
IMG_SIZE = 150

def predict_image(img_path):
    # Load ảnh và resize
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # batch_size=1

    # Dự đoán
    predictions = model.predict(img_array)  # Output shape (1,3)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]

    predicted_class = class_names[predicted_index]

    print(f"Ảnh: {img_path}")
    print(f"✅ Dự đoán: {predicted_class.upper()} ({confidence*100:.2f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("⚠️  Vui lòng truyền đường dẫn ảnh cần dự đoán.")
    else:
        predict_image(sys.argv[1])
