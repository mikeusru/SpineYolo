# from spine_preprocessing.collect_spine_data import SpineImageDataPreparer
#
# app = SpineImageDataPreparer()
# app.create_dataframe()
# app.run()
#

from spine_yolo import SpineYolo
from yolo_argparser import YoloArgparse

argparser = YoloArgparse()
args = argparser.parse_args()
app = SpineYolo(args)
app.detect()

