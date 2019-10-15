from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
import os

output_file_float32 = "./arrays/arrays_float32.h"
output_file_fix16 = "./arrays/arrays_fix16.h"

with open("keras_mnist_DCAE/keras_mnist_DCAE.json") as jfile:
    if os.path.isdir("./arrays") is False:
            os.mkdir("./arrays")

    model = load_model("keras_mnist_DCAE/keras_mnist_DCAE.h5")
    model.summary()
    model_weights_itr = model.get_weights()
    model_arrays_itr = json.load(jfile, object_pairs_hook=OrderedDict)

    arrays_fix16 = []
    arrays_float32 = []

    itr_counter = {"MaxPooling2D" : 0, "UpSampling2D" : 0, "Conv2D" : 0, "Padding2D" : 0}
    array_shapes = np.array([0, 0, 0], dtype=np.uint16)  # depth height width

    for layers in model_arrays_itr["config"]["layers"]:

        layer_name = str(layers["class_name"] + "_{}").format(itr_counter[layers["class_name"]])
        # print(layer_name)

        if layers["class_name"].find("MaxPooling2D") != -1:
            size = (layers["config"]["pool_size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] / size[0], array_shapes[2] / size[1]], dtype=np.uint16)
            itr_counter["MaxPooling2D"] += 1

        elif layers["class_name"].find("UpSampling2D") != -1:
            size = (layers["config"]["size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] * size[0], array_shapes[2] * size[1]], dtype=np.uint16)
            itr_counter["UpSampling2D"] += 1

        elif layers["class_name"].find("Conv2D") != -1:
            if itr_counter["Conv2D"] == 0:
                input_shapes = (layers["config"]["batch_input_shape"])[1:3]
            else:
                input_shapes = array_shapes[1:3]
            kernel_shapes = (layers["config"]["kernel_size"])[:]
            strides = (layers["config"]["strides"])[:]
            if layers["config"]["padding"] == "same":
                out_shapes_height = input_shapes[0] / strides[0]
                out_shapes_width = input_shapes[1] / strides[0]

                # calc padding_length(not half length)
                padding = np.array([np.max(kernel_shapes[0]-strides[0], 0), np.max(kernel_shapes[1]-strides[1], 0)])

                # generate padding layers
                # write to float32.c file

                # print(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};").format(itr_counter["Padding2D"], array_shapes[0], itr_counter["Padding2D"], array_shapes[1], itr_counter["Padding2D"], array_shapes[2]))
                # print(str("float " + layer_name + "_array[{}][{}][{}];\r\n").format(array_shapes[0], out_shapes_height + padding[0], out_shapes_height + padding[1]))
                arrays_float32.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\r\n").format(itr_counter["Padding2D"], array_shapes[0], itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_height + padding[1])))
                arrays_float32.append(str("float Padding2D_{}_array[{}][{}][{}];\r\n\r\n").format(itr_counter["Padding2D"], array_shapes[0], int(out_shapes_height + padding[0]), int(out_shapes_height + padding[1])))

                # write to fix16.c file

                # print(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};").format(itr_counter["Padding2D"], array_shapes[0], itr_counter["Padding2D"], array_shapes[1], itr_counter["Padding2D"], array_shapes[2]))
                # print(str("int16_t " + layer_name + "_array[{}][{}][{}];\r\n").format(array_shapes[0], out_shapes_height + padding[0], out_shapes_height + padding[1]))
                arrays_fix16.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\r\n").format(itr_counter["Padding2D"], array_shapes[0], itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_height + padding[1])))
                arrays_fix16.append(str("int16_t Padding2D_{}_array[{}][{}][{}];\r\n\r\n").format(itr_counter["Padding2D"], array_shapes[0], int(out_shapes_height + padding[0]), int(out_shapes_height + padding[1])))

                itr_counter["Padding2D"] += 1

            else:
                out_shapes_height = (input_shapes[0] - kernel_shapes[0]) / strides[0] + 1
                out_shapes_width = (input_shapes[1] - kernel_shapes[1]) / strides[1] + 1

            itr_counter["Conv2D"] += 1
            # depth first, not last
            array_shapes = np.array([layers["config"]["filters"], out_shapes_height, out_shapes_width], dtype=np.uint16)

        else:
            print("This Layer is not available OTL")

        # write to float32.c file

        # print(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        # print(str("float " + layer_name + "_array[{}][{}][{}];\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_float32.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_float32.append(str("float " + layer_name + "_array[{}][{}][{}];\r\n\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

        # write to fix16.c file

        # print(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        # print(str("int16_t " + layer_name + "_array[{}][{}][{}];\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_fix16.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_fix16.append(str("int16_t " + layer_name + "_array[{}][{}][{}];\r\n\r\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

        # print(itr_counter)
        # print(array_shapes.dtype, array_shapes.shape, array_shapes)
        # params_header_name_float.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h")
        # params_header_name_fix.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h")
    # print(arrays_fix16)
    # print(arrays_float32)
    with open(output_file_float32, "w") as f:
        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        f.write(" *\r\n")
        f.write(" */\r\n")
        f.write("#pragma once\r\n")
        f.write("#include <stdint.h>\r\n\r\n")
        # f.write("int main(void){\r\n")
        for i in arrays_float32:
            f.write(i)
        # f.write("}\r\n")
    with open(output_file_fix16, "w") as f:
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        f.write(" *\r\n")
        f.write(" */\r\n")
        f.write("#pragma once\r\n")
        f.write("#include <stdint.h>\r\n\r\n")
        # f.write("int main(void){\r\n")
        for i in arrays_fix16:
            f.write(i)
        # f.write("}\r\n")
