from __future__ import print_function
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img
from collections import OrderedDict
import json
import datetime
import numpy as np
import time
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("image_data_format : " + K.image_data_format())

# x_train = x_train.astype('uint8')
# x_test = x_test.astype('uint8')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = load_model("keras_mnist_DCAE.h5")
model.summary()

# x_test = x_test.astype('uint8')
# input_img = np.zeros(x_test[0:1].shape)
input_img = x_test[0:1]

times = 1000
time_array = np.zeros((times,))

for i in range(times):
    start_time = time.perf_counter()
    predict_img = model.predict(input_img)
    end_time = time.perf_counter() - start_time
    time_array[i] = end_time

np.savetxt("keras_mnist_DCAE_time_output.tsv", time_array, delimiter='\t', fmt="%25.4f")

print("end_time : ", np.sum(time_array) / times)
y = time_array
x = np.linspace(0, times, times)
plt.plot(x, y)
plt.show()
# print(input_img[0].transpose(2, 0, 1))
# print(predict_img[0].transpose(2, 0, 1))

# input_img = x_test[0:1]
# predict_img = model.predict(input_img)
input_img = input_img[0]
predict_img = predict_img[0]
save_img("keras_mnist_DCAE_input.png", input_img)
save_img("keras_mnist_DCAE_output.png", predict_img)
input_img = input_img.transpose(2, 0, 1)
predict_img = predict_img.transpose(2, 0, 1)
np.savetxt("keras_mnist_DCAE_input.tsv", input_img[0, :, :], delimiter='\t', fmt="%25.20f")
np.savetxt("keras_mnist_DCAE_output.tsv", predict_img[0, :, :], delimiter='\t', fmt="%25.20f")
