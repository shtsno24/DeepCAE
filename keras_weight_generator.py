from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
from keras_mnist_DCAE import float2fixed
import os


fractal = 16 - 2


def write_array_1D(array, f):
    f.write("{")
    for length in range(array.shape[0]):
        if(str(array.dtype).find("float") == -1):
            f.write(str("{:5d}").format(int(array[length])))
        else:
            f.write(str("{0:.20f}").format(array[length]))
        if length < array.shape[0] - 1:
            f.write(", ")
    f.write("}")


def write_array_ND(array, f):
    if len(array.shape) > 1:
        f.write("{")
        for length in range(array.shape[0]):
            write_array_ND(array[length], f)
            if length < array.shape[0] - 1:
                f.write(",\n")
                if len(array.shape) % 3 == 0 or len(array.shape) % 4 == 0:
                    f.write("\n")
        f.write("}")

    else:
        write_array_1D(array, f)


def write_weight_Conv2D_c(weight, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed = False, fractal_width = 0, array_type = None):
    # Weight generated by Keras has 4D array with (height, width, input_depth, output_depth)
    # Transpose weight axis from (height, width, input_depth, output_depth) to (output_depth, input_depth ,height, width)

    with open(file_name, 'w') as f:

        # reshape weight array
        weight = weight.transpose(3, 2, 0, 1)
        if isFixed is True:
            weight = float2fixed.float2fixed_array(array_type, fractal_width, weight)
            bias = float2fixed.float2fixed_array(array_type, fractal_width, bias)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        if isFixed is True:
            f.write(str(" * array_type : {}\n").format(weight.dtype))
            f.write(str(" * fractal_width : {} bit\n").format(fractal_width))
            f.write(str(" * bit_width : {} bit\n").format(str(8 * np.dtype(array_type).itemsize)))
        f.write(" *\n */\n")

        # include <stdint.h>
        f.write(str("#pragma once\n"))
        f.write(str("#include <stdint.h>\n\n"))

        # define data_width
        if isFixed is True:
            f.write(str("#define data_width_{} {}\n").format(str(weight_array_name[:-2]), str(8 * np.dtype(array_type).itemsize)))
            f.write(str("#define fractal_width_{} {}\n\n").format(str(weight_array_name[:-2]), str(fractal_width)))

        # weights
        f.write(str("const uint16_t shape_{}_w[] = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]))

        f.write("const " + type_name + " " + weight_array_name)
        f.write(str("[{}][{}][{}][{}] =\n").format(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]))

        write_array_ND(weight, f)
        f.write(";")
        f.write("\n\n")

        # bias
        f.write(str("const uint16_t shape_{}_b = {};\n").format(str(weight_array_name[:-2]), bias.shape[0]))

        f.write("const " + type_name + " " + bias_array_name)
        f.write("[%d] = " % bias.shape)

        write_array_ND(bias, f)
        f.write(";")
        f.write("\n")


if __name__ == "__main__":
    language = ["c", "cpp"]
    for lang in language:
        if os.path.isdir("./weights_" + lang) is False:
            os.mkdir("./weights_" + lang)

    with open("keras_mnist_DCAE/keras_mnist_DCAE.json") as jfile:
        model = load_model("keras_mnist_DCAE/keras_mnist_DCAE.h5")
        model.summary()
        model_weights_itr = model.get_weights()
        model_layers_itr = json.load(jfile, object_pairs_hook=OrderedDict)

        params_header_name_fix = []
        params_header_name_float = []
        itr_counter = 0
        for layers in model_layers_itr["config"]["layers"]:
            print()
            print(layers["class_name"])
            if(layers["class_name"].find("UpSampling") == -1 and layers["class_name"].find("MaxPooling2D") == -1 and layers["class_name"].find("UpSampling2D") == -1):
                print(layers["config"]["batch_input_shape"])
                print(layers["config"]["filters"])
                param_w = model_weights_itr[itr_counter]
                param_b = model_weights_itr[itr_counter + 1]
                itr_counter += 2
                print("weight", param_w.shape, len(param_w.shape))
                print("bias", param_b.shape, len(param_b.shape))
                write_weight_Conv2D_c(param_w,
                                    param_b,
                                    "weights_c/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_float32.h",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_w",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_b",
                                    "float")
                # params_header_name_float.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h")
                params_header_name_float.append(layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_float32.h")
                write_weight_Conv2D_c(param_w,
                                    param_b,
                                    "weights_c/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fix16.h",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_w",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_b",
                                    "int16_t", isFixed=True, fractal_width=fractal, array_type=np.int16)
                # params_header_name_fix.append("weights/" + layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h")
                params_header_name_fix.append(layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fix16.h")
            else:
                print("This Layer has no Parameter")

    with open("./weights_c/weights_float32.h", "w") as f:
        # print(params_header_name_float)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n */\n")

        for name in params_header_name_float:
            f.write('#include "' + name + '"\n')

    with open("./weights_c/weights_fix16.h", "w") as f:
        # print(params_header_name_fix)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n */\n")

        for name in params_header_name_fix:
            f.write('#include "' + name + '"\n')
