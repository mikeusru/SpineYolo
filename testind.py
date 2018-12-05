# from spine_preprocessing.collect_spine_data import SpineImageDataPreparer
#
# app = SpineImageDataPreparer()
# app.create_dataframe()
# app.run()
#

#test training w/ different amounts of training data
import os
from spine_yolo import SpineYolo
from yolo_argparser import YoloArgparse

argparser = YoloArgparse()
args = argparser.parse_args()
app = SpineYolo(args)
model_path = 'logs//training_samples_14//ep015-loss1501.284-val_loss2178.313.h5'
# model_path = os.path.join('model_data', 'yolo.h5')
app.set_starting_model_path(model_path)
app.train_yolo(training_data_to_use=2**-9)

