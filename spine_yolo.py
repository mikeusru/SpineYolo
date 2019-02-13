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
        self.model_path = os.path.expanduser(_args.model_path)
        self.log_dir = os.path.join('logs', '000')
        self.yolo_detector = None

    def detect_input_images(self):
        while True:
            img_path = input('Input image or image list filename:')
            if os.path.splitext(img_path)[1] == '.txt':
                self.detect_images_from_file_list(img_path)
                break
            try:
                image = Image.open(img_path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = self.yolo_detector.detect_image(image)
                r_image.show()
        self.yolo_detector.close_session()

    def detect(self):
        self.yolo_detector = YOLO(**{"model_path": self.model_path})
        self.detect_input_images()

    def detect_images_from_file_list(self, img_path):
        with open(img_path) as f:
            lines = f.readlines()
        for line in lines:
            try:
                line_list = line.strip().split()
                img_file = line_list[0]
                if 'Scale' in line_list:
                    scale_ind = line_list.index('Scale') + 1
                    scale = float(line_list[scale_ind])
                    image_list, coordinate_list = self.scale_and_make_sliding_windows(img_file, scale)
                    box_list = []
                    for image, relative_coordinates in zip(image_list, coordinate_list):
                        _, boxes = self.yolo_detector.detect_image(image)
                        boxes = self.shift_boxes_using_relative_coordinates(boxes, relative_coordinates)
                        box_list += boxes
                    r_image = self.put_boxes_on_image(img_file)
                    r_image.show()
                else:
                    image = Image.open(img_file)
                    r_image = self.yolo_detector.detect_image(image)
                    r_image.show()
            except:
                print('Couldn''t load image file: {}'.format(img_file))
                continue

    def scale_and_make_sliding_windows(self, img_file, scale):
        pass

    def shift_boxes_using_relative_coordinates(self, boxes, relative_coordinates):
        pass

    def put_boxes_on_image(self, img_file):
        pass

    def train_yolo(self, training_data_to_use=1):
        parsed_training_data = get_lines_from_annotation_file(self.training_data_path)
        parsed_validation_data = get_lines_from_annotation_file(self.validation_data_path)
        training_samples = round(len(parsed_training_data) * training_data_to_use)
        parsed_training_data = parsed_training_data[:training_samples]
        if training_data_to_use != 1:
            self.set_log_dir(os.path.join('logs', 'training_samples_{}'.format(training_samples)))
        train_spine_yolo(parsed_training_data, parsed_validation_data, self.log_dir, self.classes_path,
                         self.anchors_path, self.model_path)

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def set_training_data_path(self, path):
        self.training_data_path = path

    def set_validation_data_path(self, path):
        self.validation_data_path = path

    def set_model_path(self, path):
        self.model_path = path

    def prepare_image_data(self, images_path, is_labeled=False, train_test_split=0.8):
        spine_data_preparer = SpineImageDataPreparer()
        spine_data_preparer.set_initial_directory(images_path)
        spine_data_preparer.set_labeled_state(is_labeled)
        spine_data_preparer.set_train_test_split(train_test_split)
        spine_data_preparer.run()


if __name__ == '__main__':
    argparser = YoloArgparse()
    args = argparser.parse_args()
    print(args)
    app = SpineYolo(args)
    next_step = input('detect or train? : ')
    if next_step == 'detect':
        app.detect()
    if next_step == 'train':
        app.train_yolo()
