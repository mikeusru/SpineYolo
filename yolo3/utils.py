"""Miscellaneous utility functions."""

from functools import reduce
from data_aug.data_aug import *
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def show_image_transformation(img, box):
    if False:
        print('box size: {}'.format(box.size))
        plt.imshow(draw_rect(img, box))
        plt.show()


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def do_data_augmentation(annotation_line, input_shape, max_boxes=20):
    """using https://github.com/Paperspace/DataAugmentationForObjectDetection"""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image)

    # correct boxes
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes: box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy

    show_image_transformation(image_data, box)

    box_exists = False
    count = 0

    if np.random.rand() > .5:
        while box_exists is False:
            count += 1
            if count > 100:
                print('counter reached {}'.format(count))
            image_data_transformed, box_transformed = RandomHorizontalFlip(.5)(image_data.copy(), box.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomContrastStretch(.3)(image_data.copy(), box.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomHistogramEqualization(.3)(image_data.copy(), box.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomAdaptiveHistogramEqualization(.3)(image_data.copy(), box.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomScale(.3, diff=True)(image_data_transformed.copy(),
                                                                                 box_transformed.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomTranslate(.3, diff=True)(image_data_transformed.copy(),
                                                                                     box_transformed.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue
            image_data_transformed, box_transformed = RandomRotate(20)(image_data_transformed.copy(),
                                                                       box_transformed.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) == 0:
                continue

            image_data_transformed, box_transformed = RandomShear(.2)(image_data_transformed.copy(), box_transformed.copy())
            # show_image_transformation(image_data_transformed, box_transformed)
            if len(box_transformed) > 0:
                box_exists = True
    else:
        box_transformed = box.copy()
        image_data_transformed = image_data.copy()
    box_data = np.zeros((max_boxes, 5))
    box_data[:len(box_transformed)] = box_transformed
    image_data_transformed = np.array(image_data_transformed) / 255.
    show_image_transformation(image_data_transformed, box_data)

    return image_data_transformed, np.round(box_data, 0)


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
