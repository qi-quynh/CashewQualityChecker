from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import os
import numpy as np

# Đường dẫn thư mục gốc (chứa nguyen/ vo/ hu/)
input_base_path = 'dataset/'
output_base_path = 'augmented_dataset/'

# Cấu hình tăng dữ liệu
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Tạo ảnh tăng cường cho mỗi ảnh trong từng lớp
for class_name in os.listdir(input_base_path):
    input_class_dir = os.path.join(input_base_path, class_name)
    output_class_dir = os.path.join(output_base_path, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(input_class_dir):
        img_path = os.path.join(input_class_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Tạo 10 ảnh mới từ mỗi ảnh gốc
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_dir,
                                  save_prefix=class_name, save_format='jpg'):
            i += 1
            if i >= 10:
                break

print("✅ Đã tăng dữ liệu thành công.")
