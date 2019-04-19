import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename

from spine_yolo import SpineYolo
from yolo_argparser import YoloArgparse
from prepare_image_data import prepare_image_data

LARGE_FONT = "Verdana, 12"
NORMAL_FONT = "Verdana, 10"


class SpineYoloGui(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "Spine Yolo")
        tk.Tk.geometry(self, newGeometry='1000x600+200+200')
        self.spine_yolo = SpineYolo(YoloArgparse().parse_args())
        self.training_data_path = tk.StringVar()
        self.trained_model_path = tk.StringVar()
        self.train_test_split = tk.StringVar()
        self.log_dir = tk.StringVar()
        self.image_data_out_path = None
        self.import_settings()
        self.gui = self.define_gui_elements()

    def define_gui_elements(self):
        gui = dict()
        gui['training_data_folder_label'] = tk.Label(self, text="Training Data Folder:",
                                                     font=LARGE_FONT)
        gui['training_data_folder_label'].grid(row=0, column=0, sticky='nw', padx=10, pady=10)
        gui['select_training_data_button'] = tk.Button(self, text="...",
                                                       font=NORMAL_FONT,
                                                       command=self.select_training_data)
        gui['select_training_data_button'].grid(row=0, column=2, sticky='nw', padx=10, pady=10)

        gui['prepare_training_data_button'] = tk.Button(self, text="Prepare",
                                                        font=LARGE_FONT,
                                                        command=self.prepare_training_data)
        gui['prepare_training_data_button'].grid(row=0, column=3, sticky='nw', padx=10, pady=10)

        gui['training_data_folder_preview_entry'] = ttk.Entry(self,
                                                              width=80,
                                                              textvariable=self.training_data_path,
                                                              state='readonly')
        gui['training_data_folder_preview_entry'].grid(row=0, column=1, padx=10, pady=10, sticky='nw')
        gui['model_path_label'] = tk.Label(self, text="Trained Model File:",
                                           font=LARGE_FONT)
        gui['model_path_label'].grid(row=1, column=0, sticky='nw', padx=10, pady=10)
        gui['select_model_button'] = tk.Button(self, text="...",
                                               font=NORMAL_FONT,
                                               command=self.select_model)
        gui['select_model_button'].grid(row=1, column=2, sticky='nw', padx=10, pady=10)
        gui['model_path_entry'] = ttk.Entry(self,
                                            width=80,
                                            textvariable=self.trained_model_path,
                                            state='readonly')
        gui['model_path_entry'].grid(row=1, column=1, padx=10, pady=10, sticky='nw')
        gui['train_test_split_label'] = tk.Label(self, text="Trained Model File:",
                                                 font=LARGE_FONT)
        gui['train_test_split_label'].grid(row=1, column=0, sticky='nw', padx=10, pady=10)

        gui['train_model_button'] = tk.Button(self, text='Train Model',
                                              font=LARGE_FONT,
                                              command=self.train)
        gui['train_model_button'].grid(row=3, column=0, sticky='nw', padx=10, pady=10)
        gui['detect_spines_button'] = tk.Button(self, text='Detect Spines',
                                                font=LARGE_FONT,
                                                command=self.detect_spines)
        gui['detect_spines_button'].grid(row=4, column=0, sticky='nw', padx=10, pady=10)
        gui['train_test_split_label'] = tk.Label(self, text="Train/Test Split (0-1):",
                                                 font=LARGE_FONT)
        gui['train_test_split_label'].grid(row=5, column=0, sticky='nw', padx=10, pady=10)
        gui['train_test_split_entry'] = ttk.Entry(self,
                                                  width=4,
                                                  textvariable=self.train_test_split)
        gui['train_test_split_entry'].grid(row=5, column=1, padx=10, pady=10, sticky='nw')
        return gui

    def select_training_data(self):
        path = askdirectory(initialdir=self.training_data_path.get(),
                            title="Choose a directory")
        self.training_data_path.set(path)

    def prepare_training_data(self):
        prepare_image_data(self.training_data_path.get(),
                           is_labeled=True,
                           train_test_split=self.train_test_split.get(),
                           image_data_out_path=self.image_data_out_path)

    def select_model(self):
        path = askopenfilename(initialdir=self.trained_model_path.get(),
                               title="Choose a pre-trained model")
        self.trained_model_path.set(path)

    def set_training_log_dir(self):
        path = askdirectory(initialdir=self.log_dir.get(),
                            title="Choose log file directory")
        self.log_dir.set(path)

    def detect_spines(self):
        self.spine_yolo.set_model_path(self.trained_model_path.get())
        self.spine_yolo.detect()

    def train(self):
        self.spine_yolo.set_training_data_path(os.path.join(self.training_data_path.get(), 'train.txt'))
        self.spine_yolo.set_validation_data_path(os.path.join(self.training_data_path.get(), 'validation.txt'))
        self.spine_yolo.set_model_path(self.trained_model_path.get())
        self.spine_yolo.set_log_dir(self.log_dir.get())
        self.spine_yolo.train_yolo()

    def import_settings(self):
        settings_dict = load_settings_file('settings.txt')
        self.image_data_out_path = settings_dict['image_data_out_path']
        self.training_data_path.set(settings_dict['training_data_path'])
        self.trained_model_path.set(settings_dict['trained_model_path'])
        self.train_test_split.set(settings_dict['train_test_split'])
        self.log_dir.set(settings_dict['log_dir'])


def load_settings_file(filepath='settings.txt'):
    with open(filepath) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    settings_dict = {}
    for line in lines:
        settings_dict[line[0]] = line[2]
    return settings_dict


if __name__ == "__main__":
    app = SpineYoloGui()
    try:
        app.mainloop()
    except(KeyboardInterrupt, SystemExit):
        raise
