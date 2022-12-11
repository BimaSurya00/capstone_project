from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Keras
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = './model/model-facemask.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()         


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224,224))

    x = tf.keras.utils.img_to_array(img)
    x = np.array([x])

    preds = model.predict(x)
    classes = ['With_mask', 'Without_mask']  
    predict = classes[np.argmax(preds, axis=1)[0]] 
    return predict

 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'image', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        result = str(preds)               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)