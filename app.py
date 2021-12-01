import tkinter as tk
import  cv2 as cv
import  PIL.Image, PIL.ImageTk
import os
import Camera
from keras.preprocessing import image
import numpy as np
from tensorflow import keras



class App:

    def __init__(self, window=tk.Tk(),window_name ="Face Mask Detection"):
        self.window = window
        self.window_title = window_name

        self.model = keras.models.load_model('my_model')
        self.model.load_weights("weights.h5")

        self.auto_predict = False

        self.camera = Camera.Camera()

        self.init_gui()

        self.delay = 15

        self.update()

        self.window.attributes('-topmost', True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="Prediction")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)


    def auto_predict_toggle(self):

        self.auto_predict = not self.auto_predict



    def reset(self):

        self.model = keras.models.load_model('my_model')
        self.model.load_weights("weights.h5")

        self.class_label.config(text='CLASS')

    def update(self):
        if self.auto_predict:
            self.predict()

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)
            self.window.after(self.delay, self.update)

    def predict(self):

        image_width = 150
        image_height = 150
        img = image.load_img("frame.jpg", target_size=(image_width, image_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = self.model.predict([x])
        prediction = int(classes[0][0])
        print(prediction)


        if prediction == 1:
            self.class_label.config(text="Without Mask")
            return "Without Mask"
        else:
            self.class_label.config(text="With Mask")
            return "With Mask"