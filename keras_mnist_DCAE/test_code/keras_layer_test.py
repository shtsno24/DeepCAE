from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, DepthwiseConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import save_img, array_to_img
import numpy as np
import json

# input image dimensions
img_h, img_w = 16, 12

# the data, split between train and test sets
input_shape = (6, img_h, img_w)
input_shape_keras = (img_h, img_w, 6)

model = Sequential()
model.add(DepthwiseConv2D(kernel_size=(3, 5),
                          activation='relu',
                          padding='valid',
                          input_shape=input_shape_keras))

model.build()
# model.compile(loss=keras.losses.mean_squared_error,
#             optimizer=keras.optimizers.Adadelta(),
#             metrics=['accuracy'])

model.summary()

weights = model.get_weights()
conv_w = np.ones(weights[0].shape)
conv_b = np.zeros(weights[1].shape)
input_img = np.zeros(input_shape)

conv_w = conv_w.transpose(3, 2, 0, 1)  # from(height, width, in_depth, out_depth) to (out_depth, in_depth, height, width)
for o_d in range(1):
    for i_d in range(6):
        for h in range(2):
            for w in range(1):
                conv_w[o_d][i_d][h][w] = (w + 1)
conv_w = conv_w.transpose(2, 3, 1, 0)

for l in range(3):
    for i in range(8):
        for j in range(6):
            input_img[l][i][j] = (i + 1) * (j + 1)

input_img_keras = input_img.transpose(1, 2, 0)
input_img_keras = input_img_keras.reshape((1,) + input_shape_keras)
print("weights[0].shape : ", weights[0].shape)
print("weights[1].shape : ", weights[1].shape)
print("input_img.shape : ", input_img.shape)
print("input_img_keras.shape : ", input_img_keras.shape)

test_weights = []
test_weights.append(conv_w)
test_weights.append(conv_b)
model.set_weights(test_weights)

output_img_keras = model.predict(input_img_keras)
output_img = output_img_keras.transpose(0, 3, 1, 2)

# print(input_img)
# print(input_img_keras)
# print(output_img_keras)
# print(output_img)

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
        print("depth_multiplier : ", mj["config"]["depth_multiplier"])


with open("keras_layer_test.json", "w") as f:
    json.dump(model_json, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))