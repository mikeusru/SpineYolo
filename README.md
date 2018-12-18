# SpineYolo


## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) for detecting dendritic spines. Inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K) and [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

---

## Installation (Windows)

You need an NVIDIA GPU to run this

If you don't know what you're doing, I suggest installing PyCharm and setting up a virtual environment

1. Install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
1. Install [cuDNN v7.0.xx for CUDA 9.0](https://developer.nvidia.com/rdp/cudnn-archive)
1. Clone this repository
1. Download pre-trained [SpineYoloV3 weights](https://cloud.mpfi.org/url/yolov3spinesh5) file and place it in the model_data folder
1. Install requirements from requirements.txt
1. Run SpineYolo
```
pip install -r requirements.txt
python main.py
```

## Installation (Linux)

You use linux you know what you're doing. I'll make a docker container for this eventually.

### Usage
For the Gui, run main.py

For command line, run spine_yolo. You can add in commandline arguments.
It will then ask you if you want to detect or train in the command line. 
If you type detect, you can then pass in an image file path, or a textfile containing a bunch of image file paths.

Use --help to see usage of spine_yolo.py:
```
usage: spine_yolo.py [-h] [-t TRAIN_DATA_PATH] [-v VAL_DATA_PATH]
                     [-m MODEL_PATH] [-a ANCHORS_PATH] [-c CLASSES_PATH]

Retrain or 'fine-tune' a pretrained YOLOv3 model for your own data.

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN_DATA_PATH, --train_data_path TRAIN_DATA_PATH
                        path to training data
  -v VAL_DATA_PATH, --val_data_path VAL_DATA_PATH
                        path to training data
  -m MODEL_PATH, --model_path MODEL_PATH
                        path for starting weights
  -a ANCHORS_PATH, --anchors_path ANCHORS_PATH
                        path to anchors file, defaults to yolo_anchors.txt
  -c CLASSES_PATH, --classes_path CLASSES_PATH
                        path to classes file, defaults to spine_classes.txt
```
---

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

3. Run main.py, set the model you want to use, and start training.  
    `python main.py `  
    Use your trained weights or checkpoint weights 
    
If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
