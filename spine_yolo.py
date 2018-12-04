"""
This is a class for training and evaluating yadk2
"""
import os

import colorsys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Lambda
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from data_generator import DataGenerator
from spine_preprocessing.collect_spine_data import SpineImageDataPreparer
from spine_preprocessing.spine_preprocessing import process_data
from yolo3.draw_boxes import draw_boxes
from yolo3.model import (yolo_body, yolo_eval, tiny_yolo_body)
from yolo3.utils import letterbox_image
from yolo_argparser import YoloArgparse
from yolo import YOLO
from train import train_spine_yolo


# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

IMAGE_INPUT = Input(shape=(416, 416, 3))
BOXES_INPUT = Input(shape=(None, 5))


class SpineYolo(object):

    def __init__(self, _args):
        self.train_data_path = os.path.expanduser(_args.train_data_path)
        self.validation_data_path = os.path.expanduser(_args.val_data_path)
        self.classes_path = os.path.expanduser(_args.classes_path)
        self.anchors_path = os.path.expanduser(_args.anchors_path)
        self.starting_model_path = os.path.expanduser(_args.starting_model_path)
        self.log_dir = os.path.join('logs/000')
        self.class_names = self._get_classes()
        self.anchors = self._get_anchors()
        self.input_shape = (416, 416)
        self.gpu_num = 1
        self.score = 0.5
        self.iou = 0.45
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

    def train_yolo(self):
        train_spine_yolo(self.train_data_path, self.validation_data_path, self.log_dir, self.classes_path, self.anchors_path)

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def set_data_path(self, data_path=None):
        self.data_path = data_path

    def set_starting_model_path(self, path):
        self.starting_model_path = path

    def load_yolo_model(self):
        self.model = load_model(self.starting_model_path, compile=False)

    def prepare_image_data(self, images_path, is_labeled=False):
        spine_data_preparer = SpineImageDataPreparer()
        spine_data_preparer.run()

    def _get_classes(self):
        """loads the classes"""
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """loads the anchors from a file"""
        if os.path.isfile(self.anchors_path):
            with open(self.anchors_path) as f:
                anchors = f.readline()
                anchors = [float(x) for x in anchors.split(',')]
                anchors = np.array(anchors).reshape(-1, 2)
        else:
            Warning("Could not open anchors file, using default.")
            anchors = YOLO_ANCHORS
        return anchors

    def train(self):
        """
        retrain/fine-tune the model

        logs training with tensorboard

        saves training weights in current directory

        best weights according to val_loss is saved as trained_stage_3_best.h5
        """
        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        first_round_weights = self.starting_model_path
        self.create_model(freeze_body=2, weights_path=first_round_weights)
        self.model.compile(
            optimizer=Adam(lr=1e-3), loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        params = {'dim': self.input_shape,
                  'batch_size': 32,
                  'n_classes': 1,
                  'n_channels': 3,
                  'shuffle': True}

        training_generator, validation_generator = self.make_data_generators(params)

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=6,
                                 epochs=10,
                                 callbacks=[logging, checkpoint])

        self.model.save_weights(self.get_model_file(1))
        self.draw(image_set='validation', out_path="output_images_stage_1", save_all=False)

        # unfreeze
        for i in range(len(self.model.layers)):
            self.model.layers[i].trainable = True

        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.\n')

        params = {'dim': self.input_shape,
                  'batch_size': 16,
                  'n_classes': 1,
                  'n_channels': 3,
                  'shuffle': True}

        training_generator, validation_generator = self.make_data_generators(params)

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=4,
                                 epochs=100,
                                 initial_epoch=10,
                                 callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        self.model.save_weights(self.get_model_file('final'))

        self.draw(image_set='validation', out_path="output_images_stage_2", save_all=False)

    def make_data_generators(self, params):
        partition_train = self.partition['train']
        partition_validation = self.partition['validation']
        training_generator = DataGenerator(partition_train,
                                           anchors=self.anchors,
                                           file_list=self.file_list,
                                           **params)
        validation_generator = DataGenerator(partition_validation,
                                             anchors=self.anchors,
                                             file_list=self.file_list,
                                             **params)
        return training_generator, validation_generator

    def draw(self, image_set='validation', out_path="output_images", save_all=True):
        """
        Draw bounding boxes on image data
        """
        partition_eval = self.partition[image_set]
        # load validation data
        # only annotate 100 images max

        if len(partition_eval) > 100:
            partition_eval = np.random.choice(partition_eval, (100,))
        files_to_load = self.file_list[partition_eval]
        image_data = self.load_images_to_evaluate(files_to_load)
        image_data = self.reshape_image_data_for_eval(image_data)

        boxes, scores, classes, input_image_shape = self.create_yolo_output_variables()
        # Run prediction images.
        sess = K.get_session()
        out_path_full = os.path.join('..', 'spine_yolo', 'data', 'images', out_path)
        if not os.path.exists(out_path_full):
            os.makedirs(out_path_full)
        for i in range(len(image_data)):
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    self.model_body.input: image_data[i],
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
            print('Found {} boxes for image.'.format(len(out_boxes)))

            # Plot image with predicted boxes.
            image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                          self.class_names, out_scores)
            # Save the image:
            if save_all or (len(out_boxes) > 0):
                image = Image.fromarray(image_with_boxes)
                image.save(os.path.join(out_path_full, str(i) + '.tif'))

    def create_yolo_output_variables(self):
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.model.output, self.anchors, len(self.class_names), input_image_shape, score_threshold=0.5,
            iou_threshold=0.2)
        return boxes, scores, classes, input_image_shape

    @staticmethod
    def reshape_image_data_for_eval(image_data):
        image_data = process_data(image_data)
        image_data = np.array([np.expand_dims(image, axis=0)
                               for image in image_data])
        return image_data

    @staticmethod
    def load_images_to_evaluate(files_to_load):
        image_data = [np.load(file)['image'] for file in files_to_load]
        return image_data


if __name__ == '__main__':
    argparser = YoloArgparse()
    args = argparser.parse_args()
    app = SpineYolo(args)
