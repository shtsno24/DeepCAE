from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img, array_to_img
import numpy as np

# input image dimensions
img_h, img_w = 8, 6

# the data, split between train and test sets
input_shape = (3, img_h, img_w)
input_shape_keras = (img_h, img_w, 3)

model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape_keras,
                padding='same'))


# model.compile(loss=keras.losses.mean_squared_error,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])

weights = model.get_weights()
conv_w = np.ones(weights[0].shape)
conv_b = np.zeros(weights[1].shape)
input_img = np.zeros(input_shape)

conv_w = conv_w.transpose(3, 2, 0, 1)
for l in range(6):
    for k in range(3):
        for i in range(3):
            for j in range(3):
                conv_w[l][k][i][j] = (k + 1)
conv_w = conv_w.transpose(2, 3, 1, 0)

for l in range(3):
    for i in range(8):
        for j in range(6):
            input_img[l][i][j] = (i + 1) * (j + 1)

input_img_keras = input_img.transpose(1, 2, 0)
input_img_keras = input_img_keras.reshape((1,) + input_shape_keras)
print(weights[0].shape, weights[1].shape, input_img.shape, input_img_keras.shape)

test_weights = []
test_weights.append(conv_w)
test_weights.append(conv_b)
model.set_weights(test_weights)

output_img = model.predict(input_img_keras)

print(input_img)
print(input_img_keras)
print(output_img)
print(output_img.transpose(0, 3, 1, 2))
