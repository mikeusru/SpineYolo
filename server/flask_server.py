import os
import urllib.request
from PIL import Image
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request
from spine_yolo import SpineYolo
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    # First get the sale of the image
    sp = SpineYolo()
    sp.set_model_path('model_data/yolov3_spines_combined.h5')
    sp.detect(get_image())
    r_image = sp.r_images[0]
    r_boxes = sp.r_boxes
    return render_template("results.html", boxes=r_boxes)

def get_image():
    url = 'https://www.maxplanckflorida.org/wp-content/uploads/2018/07/Figure-press-release-01-1-300x297.jpg'
    with urllib.request.urlopen(url) as img_url:
        with open('temp.jpg', 'wb') as f:
            f.write(img_url.read())

    img = Image.open('temp.jpg')
    return img


if __name__ == '__main__':
    # print("loading iris model")
    # irisModel = load_model('iris_model.h5')
    # print("iris model loaded")
    # port = int(os.environ.get('PORT', 8888))
    app.run(host='0.0.0.0', port=8888, debug=True)
