"""
This is a class for training and evaluating yadk2
"""
import colorsys
import os

import numpy as np
from PIL import Image, ImageFont, ImageDraw
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
                r_image, _, _, _ = self.yolo_detector.detect_image(image)
                r_image.show()
        self.yolo_detector.close_session()

    def detect(self):
        self.yolo_detector = YOLO(**{"model_path": self.model_path})
        self.detect_input_images()

    def detect_images_from_file_list(self, img_path):
        with open(img_path) as f:
            lines = f.readlines()
        for line in lines:
            line_list = line.strip().split()
            img_file = line_list[0]
            if 'Scale' in line_list:
                scale_ind = line_list.index('Scale') + 1
                scale = float(line_list[scale_ind])
                spine_data_preparer = self.split_and_detect(img_file, scale)
                spine_data_preparer.dataframe_out = spine_data_preparer.dataframe_out.apply(self.shift_boxes, axis=1)
                # for row in spine_data_preparer.dataframe_out:
                #     if row.boxes.size > 0:
                #         for box, score in zip(row.boxes, row.scores):
                #             box[[0, 2]] += row.y
                #             box[[1, 3]] += row.x
                #             boxes_shifted_and_scores.append((box, score))
                r_image = self.put_boxes_on_image(img_file, spine_data_preparer.dataframe_out)
                r_image.show()
            else:
                try:
                    image = Image.open(img_file)
                    r_image, _, _, _ = self.yolo_detector.detect_image(image)
                    r_image.show()
                except:
                    print('Couldn''t load image file: {}'.format(img_file))
                    continue

    @staticmethod
    def shift_boxes(row):
        if row.out_boxes is not None:
            for i in range(len(row.out_boxes)):
                row.out_boxes[i][[0, 2]] += row.y
                row.out_boxes[i][[1, 3]] += row.x
        return row

    def split_and_detect(self, img_file, scale):
        spine_data_preparer = SpineImageDataPreparer()
        spine_data_preparer.set_detector(self.yolo_detector)
        spine_data_preparer.set_labeled_state(False)
        spine_data_preparer.set_resizing(True)
        spine_data_preparer.set_sliding_window_props(True)
        spine_data_preparer.set_saving(False)
        spine_data_preparer.run_on_single_image(img_file, scale)
        return spine_data_preparer

    def put_boxes_on_image(self, img_file, analyzed_dataframe):
        image = Image.open(img_file)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=max(np.floor(2e-2 * image.size[1] + 0.5).astype('int32'), 8))
        thickness = max((image.size[0] + image.size[1]) // 900, 1)
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x, 1., 1.)
                      for x in range(1)]
        colors = [(0, 0, 255)]
        for boxes, score in zip(analyzed_dataframe.out_boxes.values, analyzed_dataframe.scores.values):
            if boxes is not None:
                for box in boxes:
                    label = '{:.2f}'.format(score[0])
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=colors[0])
                    # draw.rectangle(
                    #     [tuple(text_origin), tuple(text_origin + label_size)],
                    #     fill=colors[0])
                    # draw.text(text_origin, label, fill=colors[0], font=font)
                    del draw
        return image

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
