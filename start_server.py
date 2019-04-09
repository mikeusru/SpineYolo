import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from spine_yolo import SpineYolo

app = Flask(__name__)
CORS(app)


@app.route("/price/", methods=['GET'])
def return_price():
    sp = SpineYolo()
    sp.set_model_path('model_data/yolov3_spines_combined.h5')
    sp.detect()
    r_image = sp.r_images[0]
    return jsonify(r_image)


@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome to SpineYolo <h1>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
