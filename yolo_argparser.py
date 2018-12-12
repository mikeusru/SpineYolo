import argparse

import os


class YoloArgparse(argparse.ArgumentParser):
    def __init__(self):
        super(YoloArgparse, self).__init__(
            description="Retrain or 'fine-tune' a pretrained YOLOv3 model for your own data.")

        self.add_argument(
            '-t',
            '--train_data_path',
            help="path to training data",
            default=os.path.join('data', 'sliding_window_images', 'train.txt'))

        self.add_argument(
            '-v',
            '--val_data_path',
            help="path to training data",
            default=os.path.join('data', 'sliding_window_images', 'validation.txt'))

        self.add_argument(
            '-s',
            '--model_path',
            help="path for starting weights",
            default=os.path.join('model_data', 'yolo_spines_scaled_data_aug.h5'))

        self.add_argument(
            '-a',
            '--anchors_path',
            help='path to anchors file, defaults to yolo_anchors.txt',
            default=os.path.join('model_data', 'yolo_anchors.txt'))

        self.add_argument(
            '-c',
            '--classes_path',
            help='path to classes file, defaults to spine_classes.txt',
            default=os.path.join('model_data', 'spine_classes.txt'))
