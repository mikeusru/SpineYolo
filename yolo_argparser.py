import argparse

import os


class YoloArgparse(argparse.ArgumentParser):
    def __init__(self):
        super(YoloArgparse, self).__init__(
            description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

        self.add_argument(
            '-d',
            '--data_path',
            help="path to numpy data file (.npz) list of all npz file paths in 'file_list' which have 'image' and 'boxes'",
            default=os.path.join('..', 'DATA', 'underwater_data.npz'))

        self.add_argument(
            '-t',
            '--train',
            help="set training to (default) 'on' or 'off'",
            default='on')

        self.add_argument(
            '-s',
            '--starting_weights',
            help="path for starting weights",
            default=os.path.join('..', 'spine_yolo', 'trained_stage_3_best.h5'))

        self.add_argument(
            '-f',
            '--from_scratch',
            help="don't load starting weights (on/off)",
            default='on')

        self.add_argument(
            '-a',
            '--anchors_path',
            help='path to anchors file, defaults to yolo_anchors.txt',
            default=os.path.join('..', 'spine_yolo', 'model_data', 'yolo_anchors.txt'))

        self.add_argument(
            '-o',
            '--overfit_single_image',
            help='test script on single image',
            default='off')

        self.add_argument(
            '-c',
            '--classes_path',
            help='path to classes file, defaults to pascal_classes.txt',
            default=os.path.join('..', 'spine_yolo', 'model_data', 'spine_classes.txt'))
