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
    itr_counter = 0
    for layers in model_layers_itr["config"]["layers"]:
        print()
        print(layers["class_name"])
        if layers["class_name"].find("UpSampling") != -1:
            print("This Layer has no Params")
        elif layers["class_name"].find("MaxPooling2D") != -1:
            print("This Layer has no Params")
        elif layers["class_name"].find("UpSampling2D") != -1:
            print("This Layer has no Params")
        else:
            print(layers["config"]["batch_input_shape"])
            print(layers["config"]["filters"])

            array_shapes_buffer = (layers["config"]["batch_input_shape"])[0:3]
            array_shapes_buffer[0] = layers["config"]["filters"]
            array_shapes = np.array(array_shapes_buffer, dtype=np.uint16)

            print(array_shapes.dtype, array_shapes.shape, array_shapes)

        # params_header_name_float.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h")
        # params_header_name_fix.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h")
