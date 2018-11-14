import os
import re
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform


class SpineImageDataPreparer:

    def __init__(self):
        self.do_sliding_windows = True
        self.labeled = True
        self.resize_to_scale = True
        self.target_scale_px_per_um = 10
        self.sliding_window_side = 256
        self.sliding_window_step = 128
        self.initial_directory = '../test'
        self.save_directory = os.path.join('..', 'spine_yolo', 'data', 'images', 'in')
        self.output_file_list = []
        self.dataframe = None

    def set_labeled_state(self, labeled):
        self.labeled = labeled

    def set_resizing(self, do_resizing):
        self.resize_to_scale = do_resizing

    def set_sliding_window_props(self, make_windows, window_side=256, window_step=128):
        self.do_sliding_windows = make_windows
        if make_windows:
            self.sliding_window_side = window_side
            self.sliding_window_step = window_step

    def set_initial_directory(self, path):
        self.initial_directory = path

    def set_save_directory(self, path=None):
        if path is not None:
            self.save_directory = path
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def run(self):
        self.create_dataframe()
        self.load_all_data()
        self.translate_all_boxes_to_yolo()
        self.rescale_all_data()
        self.convert_all_images_to_float()
        self.save_images_as_sliding_windows()

    def run_on_single_image(self, image_file):
        self.dataframe = pd.DataFrame({'img_path': [image_file]})
        self.load_all_data()
        self.convert_all_images_to_float()
        if self.do_sliding_windows:
            sliding_windows_x_shift = []
            sliding_windows_y_shift = []
            windows = []
            for (x, y, window, boxes_in_window) in self.yield_sliding_windows(self.dataframe['images']):
                sliding_windows_x_shift.append(x)
                sliding_windows_y_shift.append(y)
                windows.append(window)
            return sliding_windows_x_shift, sliding_windows_y_shift, windows
        else:
            return self.dataframe['images'][0]

    def create_dataframe(self):
        image_files, info_files, bbox_files = self.get_initial_file_lists()
        if self.labeled:
            self.dataframe = pd.DataFrame({'img_path': image_files,
                                           'info_path': info_files,
                                           'bounding_boxes_path': bbox_files})
            img_id = lambda in_path: in_path.split('\\')[-2][-6:]
        else:
            self.dataframe = pd.DataFrame({'img_path': image_files})
            img_id = lambda in_path: in_path.split('\\')[-1]
        self.dataframe['ImageID'] = self.dataframe['img_path'].map(img_id)
        return self.dataframe

    def get_initial_file_lists(self):
        info_files = []
        bbox_files = []
        if self.labeled:
            img_files = glob(os.path.join(self.initial_directory, '*', '*.tif'))
            info_files = glob(os.path.join(self.initial_directory, '*', '*.txt'))
            bbox_files = glob(os.path.join(self.initial_directory, '*', '*.csv'))
        else:
            img_files = glob(os.path.join(self.initial_directory, '*.tif'))
        return img_files, info_files, bbox_files

    @staticmethod
    def load_image(img_file):
        image = np.array(Image.open(img_file))
        dtype_max = np.iinfo(image.dtype).max
        image = (image / dtype_max * 255).astype(np.uint8)
        return image

    @staticmethod
    def read_scale(info_file):
        scale_px_per_um = 0
        lines = [line for line in open(info_file)]
        regexp_scale = re.compile("(-?[0-9.]*) px per um")
        for line in lines:
            if regexp_scale.search(line):
                scale_px_per_um = regexp_scale.search(line).group(1)
        return float(scale_px_per_um)

    @staticmethod
    def read_bounding_boxes(bbox_file):
        boxes = np.genfromtxt(bbox_file, delimiter=',')
        return boxes

    def load_all_data(self):
        self.dataframe['images'] = self.dataframe['img_path'].map(self.load_image)
        if self.labeled:
            self.dataframe['boxes'] = self.dataframe['bounding_boxes_path'].map(self.read_bounding_boxes)
            self.dataframe['scale'] = self.dataframe['info_path'].map(self.read_scale)

    @staticmethod
    def boxes_to_yolo(boxes):
        # function to set box parameters as x_center, y_center, box_width, box_height, class
        # class is 1 for all boxes
        yolo_boxes = []
        boxes = boxes.reshape(-1, 4)
        for box in boxes:
            if box.size > 0:
                box = np.append(box, 1)
                box[0] = box[0] + box[2] / 2
                box[1] = box[1] + box[3] / 2
            else:
                box.reshape(-1, 5)
            yolo_boxes.append(box)
        return np.array(yolo_boxes)

    def translate_all_boxes_to_yolo(self):
        if self.labeled:
            self.dataframe['boxes'] = self.dataframe['boxes'].map(self.boxes_to_yolo)

    def rescale_data(self, image, scale, boxes=None):
        boxes_rescaled = None
        resize_scale = self.target_scale_px_per_um / scale
        new_shape = np.array(image.shape)
        new_shape[:2] = np.array(new_shape[:2] * resize_scale, dtype=np.int)
        if boxes is not None:
            boxes_rescaled = boxes * resize_scale
        image_rescaled = transform.resize(image, new_shape)
        return image_rescaled, boxes_rescaled

    def rescale_row_of_data(self, row):
        if self.labeled:
            image, boxes = self.rescale_data(row['images'],
                                             scale=row['scale'],
                                             boxes=row['boxes'])
            row['boxes'] = boxes
        else:
            image, _ = self.rescale_data(row['images'],
                                         scale=row['scale'])
        row['images'] = image
        return row

    def rescale_all_data(self):
        if self.resize_to_scale:
            self.dataframe = self.dataframe.apply(self.rescale_row_of_data, axis=1)

    @staticmethod
    def convert_image_to_float(image):
        image_float = image.astype(np.float16) / np.max(image)
        return image_float

    def convert_all_images_to_float(self):
        self.dataframe['images'] = self.dataframe['images'].map(self.convert_image_to_float)

    def yield_sliding_windows(self, image, boxes=None):
        for y in range(0, image.shape[0], self.sliding_window_step):
            for x in range(0, image.shape[1], self.sliding_window_step):
                boxes_in_window = boxes
                if boxes is not None:
                    if boxes.size > 0:
                        boxes = boxes.reshape(-1, 5)
                        boxes_in_window_ind = (boxes[:, 0] > x) & \
                                              (boxes[:, 0] < x + self.sliding_window_side) & \
                                              (boxes[:, 1] > y) & \
                                              (boxes[:, 1] < y + self.sliding_window_side)
                        boxes_in_window = boxes[boxes_in_window_ind,]
                        boxes_in_window[:, 0] = boxes_in_window[:, 0] - x
                        boxes_in_window[:, 1] = boxes_in_window[:, 1] - y
                img_window = image[y:y + self.sliding_window_side, x:x + self.sliding_window_side]
                yield (x, y, img_window, boxes_in_window)

    def save_images_as_sliding_windows(self):
        self.output_file_list = []
        for index, row in self.dataframe.iterrows():
            if (index % 500 == 0) & (index > 1):
                print('Splitting image #{}/{}'.format(index, len(self.dataframe)))
            image_dir = self.create_image_directory(index)
            if self.do_sliding_windows:
                self.make_and_save_sliding_windows(row, image_dir)
            else:
                file_path = os.path.join(image_dir, f'image_{index}_data.npz')
                self.save_images_and_data(row, file_path)
        self.save_file_list()
        print('Saving done yay')

    def make_and_save_sliding_windows(self, row, image_dir):
        if self.labeled:
            boxes = row['boxes']
        else:
            boxes = None
        for (x, y, window, boxes_in_window) in self.yield_sliding_windows(row['images'], boxes):
                self.save_sliding_window(image_dir, x, y, window, boxes_in_window)



    def save_images_and_data(self, row, path):
        if self.labeled:
            self.write_data(path, row['images'], row['boxes'])
        else:
            self.write_data(path, row['images'])

    def create_image_directory(self, index):
        image_dir = os.path.join(self.save_directory, f'image{index}')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def save_sliding_window(self, image_dir, x, y, window, boxes_in_window):
        window_file_path = os.path.join(image_dir, f'window_x_{x}_y_{y}_data.npz')
        self.write_data(window_file_path, window, boxes_in_window)

    def write_data(self, path, image, boxes=None):
        self.output_file_list.append(path)
        if boxes is None:
            np.savez(path, image=image)
        else:
            np.savez(path, image=image, boxes=boxes)

    def save_file_list(self):
        self.output_file_list = np.array(self.output_file_list)
        np.savez(os.path.join(self.save_directory, 'file_list.npz'), file_list=self.output_file_list)
