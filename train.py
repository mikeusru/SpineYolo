"""
Retrain the YOLOv3 model for your own dataset.
"""
import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data, do_data_augmentation


def _main(parsed_training_data=None, parsed_validation_data=None, log_dir=None,
          classes_path=None, anchors_path=None, model_path=None):
    if parsed_training_data is None:
        annotation_path_train = os.path.join('data', 'sliding_window_images', 'train.txt')
        parsed_training_data = get_lines_from_annotation_file(annotation_path_train)
    if parsed_validation_data is None:
        annotation_path_validation = os.path.join('data', 'sliding_window_images', 'validation.txt')
        parsed_validation_data = get_lines_from_annotation_file(annotation_path_validation)
    if log_dir is None:
        log_dir = 'logs/000/'
    if classes_path is None:
        classes_path = 'model_data/spine_classes.txt'
    if anchors_path is None:
        anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if model_path is None:
        if is_tiny_version:
            model_path = 'model_data/tiny_yolo_weights.h5'
        else:
            model_path = 'model_data/yolo_spines_scaled.h5'
    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2,
                         weights_path=model_path)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(os.path.join(log_dir , 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    num_val = len(parsed_validation_data)
    num_train = len(parsed_training_data)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if False:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(parsed_training_data, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(parsed_validation_data, batch_size, input_shape,
                                                                   anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            max_queue_size=4,
                            workers=4,
                            epochs=20,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 8  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(parsed_training_data, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(parsed_validation_data, batch_size, input_shape,
                                                                   anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=1000,
                            initial_epoch=10,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    """create the training model, for Tiny YOLOv3"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            # image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image, box = do_data_augmentation(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def get_lines_from_annotation_file(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line for line in lines if len(line.strip()) > 0]
    lines = [line for line in lines if len(line.split()) > 1]
    lines = [line for line in lines if line.split()[1] != ',']
    return lines


if __name__ == '__main__':
    _main()


def train_spine_yolo(train_path, val_path, log_dir, classes_path, anchors_path, model_path):
    _main(train_path, val_path, log_dir, classes_path, anchors_path, model_path)
