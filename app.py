# from PIL import Image
import cv2
import numpy as np
# import pandas as pd
from flask import Flask, render_template, send_file, request
from model import classification_img
import torch

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def gambar():
    foto = request.files['inputGambar']
    filename = foto.filename
    content_type = foto.content_type
    data = foto.read()

    # path = foto.save('static/images/' + filename)

    processed_image = processed_image(data)

    if foto is not None:
        file_bytes = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        model = torch.load('best.pt')
        klasifikasi = classification_img(opencv_image, model)
    
    return render_template('index.html', hasil=klasifikasi)

@app.route('/show')
def show(filename):
    foto = request.files['inputGambar']
    filename = foto.filename

    foto.save('static/images/' + filename)
    path = ('static/images/' + filename)
    image = cv2.imread(path)

    return send_file(image)
