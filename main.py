import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename

from spine_yolo import SpineYolo
from yolo_argparser import YoloArgparse

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
        self.log_dir = tk.StringVar()
        self.new_model_path = tk.StringVar()
        self.set_default_variables()
        self.gui = self.define_gui_elements()

    def set_default_variables(self):
        default_path = os.path.expanduser('~')
        self.training_data_path.set(os.path.join(default_path, 'training_data'))
        self.log_dir.set(os.path.join(default_path, 'logs'))
        self.new_model_path.set(os.path.join(default_path, 'models'))
        self.trained_model_path.set(os.path.join(default_path, 'models', 'trained_model.h5'))

    def define_gui_elements(self):
        gui = dict()
        gui['training_data_folder_label'] = tk.Label(self, text="Training Data Folder:",
                                                     font=LARGE_FONT)
        gui['training_data_folder_label'].grid(row=0, column=0, sticky='nw', padx=10, pady=10)
        gui['select_training_data_button'] = tk.Button(self, text="...",
                                                       font=NORMAL_FONT,
                                                       command=self.select_training_data)
        gui['select_training_data_button'].grid(row=0, column=2, sticky='nw', padx=10, pady=10)
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
        gui['train_model_button'] = tk.Button(self, text='Train Model',
                                              font=LARGE_FONT,
                                              command=self.train)
        gui['train_model_button'].grid(row=3, column=0, sticky='nw', padx=10, pady=10)
        gui['test_model_button'] = tk.Button(self, text='Test Model',
                                             font=LARGE_FONT,
                                             command=self.test)
        gui['test_model_button'].grid(row=3, column=1, sticky='nw', padx=10, pady=10)
        gui['run_on_single_image_button'] = tk.Button(self, text='Run On Single Image',
                                                      font=LARGE_FONT,
                                                      command=self.test_single_image)
        gui['run_on_single_image_button'].grid(row=4, column=0, sticky='nw', padx=10, pady=10)
        return gui

    def select_training_data(self):
        path = askdirectory(initialdir=self.training_data_path.get(),
                            title="Choose a directory")
        self.training_data_path.set(path)

    def select_model(self):
        path = askopenfilename(initialdir=self.trained_model_path.get(),
                               title="Choose a pre-trained model")
        self.trained_model_path.set(path)

    def select_folder_for_newly_trained_model(self):
        path = askdirectory(initialdir=self.new_model_path.get(),
                            title="Choose a directory for new model")
        self.new_model_path.set(path)

    def set_training_log_dir(self):
        path = askdirectory(initialdir=self.log_dir.get(),
                                 title="Choose log file directory")
        self.log_dir.set(path)

    def test(self):
        pass

    def test_single_image(self):
        pass

    def train(self):
        self.select_folder_for_newly_trained_model()
        self.set_training_log_dir()
        self.spine_yolo.toggle_training(True)
        self.spine_yolo.prepare_image_data(self.training_data_path.get(), is_labeled=True)
        self.spine_yolo.set_classes()
        self.spine_yolo.set_anchors()
        self.spine_yolo.set_partition(train_validation_split=.9, ratio_of_training_data_to_use=1)
        self.spine_yolo.set_log_dir(self.log_dir.get())
        self.spine_yolo.set_model_save_path(self.new_model_path.get())
        self.spine_yolo.run()


if __name__ == "__main__":
    app = SpineYoloGui()
    try:
        app.mainloop()
    except(KeyboardInterrupt, SystemExit):
        raise
