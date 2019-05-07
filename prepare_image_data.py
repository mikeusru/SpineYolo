import argparse
import os

from misc_utils import format_path_for_os
from spine_preprocessing.collect_spine_data import SpineImageDataPreparer


def prepare_image_data(images_path, is_labeled=False, train_test_split=0.8, image_data_out_path=None):
    image_data_out_path = format_path_for_os(image_data_out_path)
    images_path = format_path_for_os(images_path)
    if not os.path.isdir(image_data_out_path):
        os.makedirs(image_data_out_path)

    spine_data_preparer = SpineImageDataPreparer()
    spine_data_preparer.set_initial_directory(images_path)
    spine_data_preparer.set_labeled_state(is_labeled)
    spine_data_preparer.set_train_test_split(train_test_split)
    spine_data_preparer.set_save_directory(image_data_out_path)
    spine_data_preparer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--input', type=str,
        help='path to unprepared images and text files'
    )

    parser.add_argument(
        '--output', type=str,
        help='output images and text files path'
    )

    parser.add_argument(
        '--train_test_split', type=float, default=0.8,
        help='train/test split'
    )

    parser.add_argument(
        '--is_labeled', default=True,
        help='is the data labeled? default = True'
    )

    args = parser.parse_args()

    prepare_image_data(args.input, args.is_labeled, args.train_test_split, args.output)
