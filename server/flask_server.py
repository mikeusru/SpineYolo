import os
from flask import Flask, render_template, request
from spine_yolo import SpineYolo
import random

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    uploaded_image_path = upload_image(request.files.getlist('file'))
    scale = int(request.form['scale'])
    print(uploaded_image_path, scale)
    sp.detect(uploaded_image_path, scale)
    r_image = sp.r_images[0]
    r_boxes = sp.r_boxes
    filename = save_image(r_image)
    print('detection done')
    return render_template("results.html", boxes=r_boxes, image_name=filename)


def upload_image(file_list):
    upload_target = os.path.join(APP_ROOT, 'static')
    print('\n\n', upload_target, '\n\n')
    print(file_list)
    if not os.path.isdir(upload_target):
        os.mkdir(upload_target)
    for file in file_list:
        filename = file.filename
        rand_number = random.randint(1, 100000)
        destination = os.path.join(upload_target,
                                   'file_' + str(rand_number) + filename)
        print(destination)
        file.save(destination)
    return destination


def save_image(image):
    rand_number = random.randint(1, 100000)
    img_name = 'r_img' + str(rand_number) + '.jpg'
    image_path = os.path.join('server', 'static', img_name)

    image.save(image_path)
    return img_name

@app.route("/submit_training_data", methods=['POST'])
def submit_training_data():
    return render_template("add_training.html")


if __name__ == '__main__':
    print("Current Working Directory ", os.getcwd())
    sp = SpineYolo()
    sp.set_model_path('model_data/yolov3_spines_combined.h5')
    sp.set_detector()
    app.run(host='0.0.0.0', port=5000, debug=False)
