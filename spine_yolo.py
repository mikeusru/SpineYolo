"""
This is a class for training and evaluating yadk2
"""
import os

import numpy as np
from PIL import Image
from keras.layers import Input

from spine_preprocessing.collect_spine_data import SpineImageDataPreparer
from train import train_spine_yolo, get_lines_from_annotation_file
from yolo import YOLO
from yolo_argparser import YoloArgparse

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

IMAGE_INPUT = Input(shape=(416, 416, 3))
BOXES_INPUT = Input(shape=(None, 5))


class SpineYolo(object):

    def __init__(self, _args):
        self.training_data_path = os.path.expanduser(_args.train_data_path)
        self.validation_data_path = os.path.expanduser(_args.val_data_path)
        self.classes_path = os.path.expanduser(_args.classes_path)
        self.anchors_path = os.path.expanduser(_args.anchors_path)
        self.starting_model_path = os.path.expanduser(_args.starting_model_path)
        self.log_dir = os.path.join('logs','000')
        self.yolo_detector = None

    def detect_input_images(self):
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = self.yolo_detector.detect_image(image)
                r_image.show()
        self.yolo_detector.close_session()

    def detect(self):
        self.yolo_detector = YOLO()
        self.detect_input_images()

    def train_yolo(self, training_data_to_use=1):
        parsed_training_data = get_lines_from_annotation_file(self.training_data_path)
        parsed_validation_data = get_lines_from_annotation_file(self.validation_data_path)
        training_samples = round(len(parsed_training_data)*training_data_to_use)
        parsed_training_data = parsed_training_data[:training_samples]
        if training_data_to_use != 1:
            self.set_log_dir(os.path.join('logs', 'training_samples_{}'.format(training_samples)))
        train_spine_yolo(parsed_training_data, parsed_validation_data, self.log_dir, self.classes_path,
                         self.anchors_path, self.starting_model_path)

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def set_training_data_path(self, path):
        self.training_data_path = path

    def set_validation_data_path(self, path):
        self.validation_data_path = path

    def set_starting_model_path(self, path):
        self.starting_model_path = path

    def prepare_image_data(self, images_path, is_labeled=False):
        spine_data_preparer = SpineImageDataPreparer()
        spine_data_preparer.run()

if __name__ == '__main__':
    argparser = YoloArgparse()
    args = argparser.parse_args()
    app = SpineYolo(args)
    next_step = input('detect or train? : ')
    if next_step == 'detect':
        app.detect_input_images()
    if next_step == 'train':
        app.train_yolo()