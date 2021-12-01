from keras.preprocessing.image import  ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
# import pickle
#
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# from os import listdir
# from os.path import isfile, join
import keras
# from keras import optimizers

image_width = 150
image_height = 150
train_data_dir = "data/"
validation_dir = "test/"
train_samples = 30
validation_samples = 5
epoches = 10
batch_size = 200
if K.image_data_format() == 'channels_first':
    input_shape = (3, image_width, image_height)
else:
    input_shape = (image_width, image_height,3)


model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr=.0001), metrics = ['accuracy'] )

train_datagen = ImageDataGenerator(rescale= 1. /255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(image_width,image_height), batch_size=batch_size, class_mode="binary")

print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')
print(validation_generator.class_indices)


history = model.fit_generator(train_generator, steps_per_epoch=train_samples, epochs=epoches, validation_data= validation_generator, validation_steps=validation_samples)


model.save("my_model")
model.save_weights("weights.h5")

print("Model Success")