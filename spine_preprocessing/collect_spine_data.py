import os
import re
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from skimage import transform
from sklearn.model_selection import train_test_split


class SpineImageDataPreparer:

    def __init__(self):
        self.do_sliding_windows = True
        self.labeled = True
        self.resize_to_scale = True
        self.target_scale_px_per_um = 15
        self.sliding_window_side = 256
        self.sliding_window_step = 128
        self.initial_directory = os.path.join("C:\\Users\\smirnovm\\Documents\\Data\\keras_yolo3_spine_training_tifs")
        self.input_files = dict()
        self.initial_image_file_dict = dict()
        self.save_directory = os.path.join('data', 'sliding_window_images')
        self.output_file_list = []
        self.dataframe = None
        self.dataframe_out = pd.DataFrame()
        self.temp_loaded_image = None
        self.original_scale = None
        self.training_fraction = 0.85

    def set_labeled_state(self, labeled):
        self.labeled = labeled

    def set_train_test_split(self, train_test_split):
        self.training_fraction = float(train_test_split)

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
        for count, (index, row) in enumerate(self.dataframe.iterrows()):
            if (count % 100 == 0) & (count > 1):
                print('Splitting image #{}/{}'.format(count, len(self.dataframe)))
            image_dir = self.create_image_directory(count)
            self.process_individual_row(row, image_dir)
        self.write_dataframe_to_file()
        print('Saving done yay')

    def process_individual_row(self, row, image_dir):
        image_path = row.name
        self.load_image(image_path)
        if 'bounding_boxes' not in row.keys():
            row.bounding_boxes = None
        row.bounding_boxes = self.nan_to_none(row.bounding_boxes)
        self.rescale_row(row)
        self.make_and_save_sliding_windows(row, image_dir)

    @staticmethod
    def nan_to_none(bounding_boxes):
        if type(bounding_boxes) != np.ndarray:
            bounding_boxes = None
        return bounding_boxes

    def rescale_row(self, row):
        scale = row.scale
        self.original_scale = scale
        bounding_boxes = row.bounding_boxes
        if self.labeled:
            self.temp_loaded_image, row.bounding_boxes = self.rescale_data(self.temp_loaded_image,
                                                                           scale=scale,
                                                                           boxes=bounding_boxes)
        else:
            self.temp_loaded_image, _ = self.rescale_data(self.temp_loaded_image,
                                                          scale=scale)

    def run_on_single_image(self, image_file):
        self.dataframe = pd.DataFrame({'img_path': [image_file]})
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
        self.get_image_file_dict()
        self.dataframe = pd.DataFrame.from_dict(self.initial_image_file_dict, orient='index')

    def get_image_file_dict(self):
        self.input_files['train'] = os.path.join(self.initial_directory, 'train.txt')
        self.input_files['validation'] = os.path.join(self.initial_directory, 'validation.txt')
        self.input_files['image_info'] = os.path.join(self.initial_directory, 'image_info.txt')
        lines = self.file_to_list(self.input_files['image_info'])
        self.create_image_file_dict(lines)
        self.set_bounding_boxes(self.input_files['train'], 'train')
        self.set_bounding_boxes(self.input_files['validation'], 'validation')

    def set_bounding_boxes(self, path, train_or_validation):
        lines = self.file_to_list(path)
        lines = [line for line in lines if line.split()[1] != ',']
        for line in lines:
            bounding_boxes = line.split()[1:]
            bounding_boxes = np.array([np.array(box.split(',')).astype(int) for box in bounding_boxes])
            image_path = os.path.join(self.initial_directory, line.split()[0])
            self.initial_image_file_dict[image_path]['bounding_boxes'] = bounding_boxes
            self.initial_image_file_dict[image_path]['train_or_validation'] = train_or_validation

    def create_image_file_dict(self, lines):
        file_dict = dict()
        for line in lines:
            image_path = os.path.join(self.initial_directory, line.split()[0])
            image_info = line.split()[1:]
            info_dict = dict()
            for key, val in zip(image_info[0::2], image_info[1::2]):
                info_dict[key.lower()] = val.lower()
            file_dict[image_path] = info_dict
        self.initial_image_file_dict = file_dict

    @staticmethod
    def file_to_list(path):
        with open(path) as f:
            lines = f.readlines()
        lines = [line for line in lines if len(line.strip()) > 0]
        return lines

    def load_image(self, img_file):
        image = np.array(Image.open(img_file))
        dtype_max = np.iinfo(image.dtype).max
        image = (image / dtype_max * 255).astype(np.uint8)
        self.temp_loaded_image = image

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

    def rescale_data(self, image, scale, boxes=None):
        if isinstance(scale, str):
            scale = float(scale)
        boxes_rescaled = boxes
        resize_scale = self.target_scale_px_per_um / scale
        new_shape = np.array(image.shape)
        new_shape[:2] = np.array(new_shape[:2] * resize_scale, dtype=np.int)
        if boxes is not None:
            boxes_rescaled[:, :4] = boxes[:, :4] * resize_scale
        image_rescaled = transform.resize(image, new_shape, preserve_range=True).astype(np.uint8)
        return image_rescaled, boxes_rescaled

    def convert_image_to_float(self):
        self.temp_loaded_image = self.temp_loaded_image.astype(np.float16) / np.max(self.temp_loaded_image)

    def yield_sliding_windows(self, image, boxes=None):
        for y in range(0, image.shape[0], self.sliding_window_step):
            for x in range(0, image.shape[1], self.sliding_window_step):
                boxes_in_window = boxes
                if boxes is not None:
                    if boxes.size > 0:
                        # boxes should be xmin, ymin, xmax, ymax, class
                        boxes_in_window_ind = (boxes[:, 0] > x) & \
                                              (boxes[:, 2] < x + self.sliding_window_side) & \
                                              (boxes[:, 1] > y) & \
                                              (boxes[:, 3] < y + self.sliding_window_side)
                        boxes_in_window = boxes[boxes_in_window_ind,]
                        boxes_in_window[:, 0] = boxes_in_window[:, 0] - x
                        boxes_in_window[:, 2] = boxes_in_window[:, 2] - x
                        boxes_in_window[:, 1] = boxes_in_window[:, 1] - y
                        boxes_in_window[:, 3] = boxes_in_window[:, 3] - y
                img_window = image[y:y + self.sliding_window_side, x:x + self.sliding_window_side]
                yield (x, y, img_window, boxes_in_window)

    def make_and_save_sliding_windows(self, row, image_dir):
        if self.labeled:
            bounding_boxes = row.bounding_boxes
        else:
            bounding_boxes = None
        for (x, y, window, boxes_in_window) in self.yield_sliding_windows(self.temp_loaded_image, bounding_boxes):
            self.save_sliding_window(image_dir, x, y, window, boxes_in_window)

    def create_image_directory(self, count):
        image_dir = os.path.join(self.save_directory, 'image{}'.format(count))
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def save_sliding_window(self, image_dir, x, y, window, boxes_in_window):
        window_file_path = os.path.join(image_dir, 'window_x_{}_y_{}_data.tiff'.format(x, y))
        row_out = pd.Series(dict(bounding_boxes=boxes_in_window, x=x, y=y, scale=self.target_scale_px_per_um,
                                 original_scale=self.original_scale)).rename(window_file_path)
        self.dataframe_out = self.dataframe_out.append(row_out)
        self.write_image(window_file_path, window)

    def write_dataframe_to_file(self):
        self.dataframe_out.bounding_boxes = self.dataframe_out.bounding_boxes.map(self.boxes_to_strings)
        first = True
        for original_scale in set(self.dataframe_out.original_scale.values):
            #group images by scale before splitting them so train/test split has good representation of each datasete
            # cluster, and keeps image sequences separated because shuffling them gets very similar data in train/val groups
            df_image_group = self.dataframe_out.loc[self.dataframe_out['original_scale'] == original_scale]
            df_train_add, df_validation_add = train_test_split(df_image_group, test_size=1 - self.training_fraction,
                                                               shuffle=False)
            if first:
                df_train, df_validation = df_train_add, df_validation_add
                first = False
            else:
                df_train = pd.concat([df_train, df_train_add])
                df_validation = pd.concat([df_validation, df_validation_add])
        # df_train, df_validation = train_test_split(self.dataframe_out, test_size=1 - self.training_fraction)
        train_path = os.path.join(self.save_directory, 'train.txt')
        validation_path = os.path.join(self.save_directory, 'validation.txt')
        with open(train_path, 'w') as f:
            for ind, row in df_train.iterrows():
                self.write_row_to_file(row, f)
        with open(validation_path, 'w') as f:
            for ind, row in df_validation.iterrows():
                self.write_row_to_file(row, f)

    def write_row_to_file(self, row, f):
        f.write('{} '.format(row.name))
        for box in row.bounding_boxes:
            f.write('{} '.format(box))
        f.write('\n')

    @staticmethod
    def boxes_to_strings(bounding_boxes):
        if type(bounding_boxes) == np.ndarray:
            box_strings = ['{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}'.format(box[0], box[1], box[2], box[3], box[4]) for box
                           in bounding_boxes]
        else:
            box_strings = ''

        return box_strings

    def write_image(self, path, image):
        image_to_write = Image.fromarray(image)
        image_to_write.save(path)


if __name__ == '__main__':
    app = SpineImageDataPreparer()
    app.run()