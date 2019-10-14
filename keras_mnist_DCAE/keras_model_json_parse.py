from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
import float2fixed
import os


with open("keras_mnist_DCAE.json") as jfile:
    model = load_model("keras_mnist_DCAE.h5")
    model.summary()
    model_weights_itr = model.get_weights()
    model_layers_itr = json.load(jfile, object_pairs_hook=OrderedDict)

    params_header_name_fix = []
    params_header_name_float = []
    itr_counter = np.array([0, 0, 0], dtype=np.uint16)  # MaxPool, UpSampling, Conv2D
    array_shapes = np.array([0, 0, 0], dtype=np.uint16)  # depth height width

    for layers in model_layers_itr["config"]["layers"]:
        print(layers["class_name"])

        if layers["class_name"].find("MaxPooling2D") != -1:
            print("This Layer has no Params")
            size = (layers["config"]["pool_size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] / size[0], array_shapes[2] / size[1]], dtype=np.uint16)

        elif layers["class_name"].find("UpSampling2D") != -1:
            print("This Layer has no Params")
            size = (layers["config"]["size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] * size[0], array_shapes[2] * size[1]], dtype=np.uint16)

        elif layers["class_name"].find("Conv2D") != -1:
            if itr_counter[2] == 0:
                input_shapes = (layers["config"]["batch_input_shape"])[1:3]
                itr_counter[2] += 1
            else:
                input_shapes = array_shapes[1:3]
            kernel_shapes = (layers["config"]["kernel_size"])[:]
            strides = (layers["config"]["strides"])[:]
            if layers["config"]["padding"] == "same":
                out_shapes_height = input_shapes[0] / strides[0]
                out_shapes_width = input_shapes[1] / strides[0]
            else:
                out_shapes_height = (input_shapes[0] - kernel_shapes[0]) / strides[0] + 1
                out_shapes_width = (input_shapes[1] - kernel_shapes[1]) / strides[1] + 1

            # depth first, not last
            array_shapes = np.array([layers["config"]["filters"], out_shapes_height, out_shapes_width], dtype=np.uint16)

        else:
            print("This Layer is not available OTL")
        print(array_shapes.dtype, array_shapes.shape, array_shapes)
        # params_header_name_float.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h")
        # params_header_name_fix.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h")
