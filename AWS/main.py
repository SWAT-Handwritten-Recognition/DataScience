from flask import Flask, request, jsonify, render_template
from load import compare_images

import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, load_img

# load a saved model
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    database_image = request.form['database_image']
    print(database_image)
    print(type(database_image))
    frontend_image = request.form['frontend_image']
    result = compare_images(database_image, frontend_image)
    return render_template('index.html', prediction_text='{}'.format(result))

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=8080)