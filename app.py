from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

model = load_model('cashew_model_3class.h5')  # model đã train với 3 lớp
IMG_SIZE = 150
class_names = ['con_vo', 'nguyen', 'vo']  # 3 lớp tương ứng

def prepare_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="Không tìm thấy file ảnh.", confidence="", image_path="")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="Bạn chưa chọn ảnh.", confidence="", image_path="")

        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        img = prepare_image(file_path)
        preds = model.predict(img)
        class_idx = np.argmax(preds)
        prediction = class_names[class_idx]
        confidence = f"{preds[0][class_idx]*100:.2f}%"
        image_path = '/' + file_path

    return render_template('index.html', prediction=prediction, confidence=confidence, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
