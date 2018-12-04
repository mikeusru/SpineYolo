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
app.set_starting_model_path(os.path.join('model_data', 'yolo.h5'))
app.train_yolo(training_data_to_use=1e-1)

