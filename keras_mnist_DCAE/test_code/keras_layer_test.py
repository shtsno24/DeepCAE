from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img, array_to_img
import numpy as np
import json

# input image dimensions
img_h, img_w = 7, 7

# the data, split between train and test sets
input_shape = (3, img_h, img_w)
input_shape_keras = (img_h, img_w, 3)

model = Sequential()
model.add(SeparableConv2D(6, kernel_size=(3, 3),
                          activation='relu',
                          padding='same',
                          input_shape=input_shape_keras))

model.build()
# model.compile(loss=keras.losses.mean_squared_error,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])

model.summary()

weights = model.get_weights()
for w in weights:
    print(w.shape, w.dtype)
conv_w_0 = np.ones(weights[0].shape)
conv_w_1 = np.ones(weights[1].shape)
conv_b = np.zeros(weights[2].shape)
input_img = np.zeros(input_shape)

conv_w_0 = conv_w_0.transpose(3, 2, 0, 1)  # from(height, width, in_depth, out_depth) to (out_depth, in_depth, height, width)
for o_d in range(1):
    for i_d in range(3):
        for h in range(3):
            for w in range(3):
                conv_w_0[o_d][i_d][h][w] = (w + 1)
conv_w_0 = conv_w_0.transpose(2, 3, 1, 0)

conv_w_1 = conv_w_1.transpose(3, 2, 0, 1)
for o_d in range(6):
    for i_d in range(3):
        for h in range(1):
            for w in range(1):
                conv_w_1[o_d][i_d][h][w] = (w + 1)
conv_w_1 = conv_w_1.transpose(2, 3, 1, 0)

for length in range(6):
    conv_b[length] = length

for d in range(3):
    for h in range(7):
        for w in range(7):
            input_img[d][h][w] = (h + 1) * (w + 1)

input_img_keras = input_img.transpose(1, 2, 0)
input_img_keras = input_img_keras.reshape((1,) + input_shape_keras)
print("weights[0].shape : ", weights[0].shape)
print("weights[1].shape : ", weights[1].shape)
print("input_img.shape : ", input_img.shape)
print("input_img_keras.shape : ", input_img_keras.shape)

test_weights = []
test_weights.append(conv_w_0)
test_weights.append(conv_w_1)
test_weights.append(conv_b)
model.set_weights(test_weights)

output_img_keras = model.predict(input_img_keras)
output_img = output_img_keras.transpose(0, 3, 1, 2)

print(input_img)
print(input_img_keras)
print(output_img_keras)
print(output_img)

print("output_img_keras.shape", output_img_keras.shape)
print("output_img.shape : ", output_img.shape)

model.save("keras_layer_test.h5")
model.save_weights("keras_layer_test_weight.h5")
model_json = model.to_json()
with open("keras_layer_test.json", "w") as f:
    f.write(model_json)

with open("keras_layer_test.json", "r") as f:
    model_json = json.load(f)
    for mj in model_json["config"]["layers"]:
        print("layer_name : ", mj["class_name"])
        print("first_input_shape : ", (mj["config"]["batch_input_shape"])[1:4])
        print("padding : ", mj["config"]["padding"])
        print("stride : ", mj["config"]["strides"])
        print("kernel_size : ", mj["config"]["kernel_size"])
        print("activation : ", mj["config"]["activation"])
        print("filters : ", mj["config"]["filters"])


with open("keras_layer_test.json", "w") as f:
    json.dump(model_json, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
