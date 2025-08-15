from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import sqlite3
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
DB_FILE = 'patients.db'
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model (update path if needed)
MODEL_PATH = 'vgg19_mobilenet_fusion.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    phone TEXT,
                    image_path TEXT,
                    prediction TEXT,
                    confidence REAL
                )''')
    conn.commit()
    conn.close()

def make_heatmap(img_array, preds, last_conv_layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_type = request.form['upload_type']

        if upload_type == 'single':
            name = request.form['name']
            phone = request.form['phone']
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)
                img_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(img_path)

                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Preprocess for both branches
                vgg_input = vgg19_preprocess(img_array.copy())
                mobilenet_input = mobilenet_preprocess(img_array.copy())

                # Predict
                preds = model.predict([vgg_input, mobilenet_input])[0][0]  # scalar value
                # For binary classification, preds ~ probability of "malignant" (usually)
                if preds < 0.5:
                    class_name = 'Benign'
                else:
                    class_name = 'Malignant'

                confidence = round((max(preds, 1 - preds) * 100), 2)


                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO patients (name, phone, image_path, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
                          (name, phone, img_path, class_name, confidence))
                conn.commit()
                conn.close()

                return render_template('result.html', name=name, phone=phone, prediction=class_name, confidence=confidence,
                                       image_path=img_path)

        elif upload_type == 'bulk':
            folder = request.files.getlist('bulk_images')
            for file in folder:
                filename = secure_filename(file.filename)
                img_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(img_path)

                try:
                    name, phone = filename.rsplit('_', 1)
                    phone = phone.split('.')[0]
                except:
                    name, phone = 'Unknown', 'Unknown'

                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                preds = model.predict(img_array)
                if preds < 0.5:
                    class_name = 'benign'
                else:
                    class_name = 'malignant'

                confidence = max(preds, 1 - preds)

                heatmap = make_heatmap(img_array, preds)
                heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + cv2.imread(img_path)
                heatmap_filename = 'heatmap_' + filename
                heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
                cv2.imwrite(heatmap_path, superimposed_img)

                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO patients (name, phone, image_path, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
                          (name, phone, img_path, class_name, confidence))
                conn.commit()
                conn.close()

            return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
