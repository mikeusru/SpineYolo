from spine_yolo import SpineYolo
from main import load_settings_file
import os

sp = SpineYolo()
settings_dict = load_settings_file('settings.txt')
sp.set_model_path(settings_dict['trained_model_path'])
sp.set_detector()
sp.yolo_detector.save_model(os.path.join('model_data','yolov3_spines_model.json'))