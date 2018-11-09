import numpy as np
from skimage import transform

#boxes = None happens when only images are preprocessed for the draw phase
def process_data(images, boxes = None):
    '''processes the data'''
    orig_size = [np.array([i.shape[1], i.shape[0]]) for i in images]

    # Image preprocessing.
    processed_images = [np.stack([i, i, i], 2) for i in images]
    processed_images = np.array([transform.resize(i, (416, 416, 3), mode='constant') for i in processed_images])

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of boxes: x_center, y_center, box_width, box_height, class.
        boxes = [box.reshape((-1, 5)) for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        # boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        # boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [box[:, :2] / o_size for box, o_size in zip(boxes, orig_size)]
        boxes_wh = [box[:, [2, 3]] / o_size for box, o_size in zip(boxes, orig_size)]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)