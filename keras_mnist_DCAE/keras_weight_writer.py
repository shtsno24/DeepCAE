from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
import float2fixed


def write_array_1D(array, f):
    f.write("{")
    for length in range(array.shape[0]):
        f.write(str(array[length]))
        if length < array.shape[0] - 1:
            f.write(", ")
    f.write("}")


def write_array_ND(array, f):
    if len(array.shape) > 1:
        f.write("{")
        for length in range(array.shape[0]):
            write_array_ND(array[length], f)
            if length < array.shape[0] - 1:
                f.write(",\r\n")
                if len(array.shape) % 3 == 0 or len(array.shape) % 4 == 0:
                    f.write("\r\n")
        f.write("}")

    else:
        write_array_1D(array, f)


def write_weight_Conv2D(weight, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed = False, fractal_width = 0, array_type = None):
    # Weight generated by Keras has 4D array with (height, width, input_depth, output_depth)
    # Transpose weight axis from (height, width, input_depth, output_depth) to (output_depth, input_depth ,height, width)

    with open(file_name, 'w') as f:

        # reshape weight array
        weight = weight.transpose(3, 2, 0, 1)
        if isFixed == True:
            weight = float2fixed.float2fixed_array(array_type, fractal_width, weight)
            bias = float2fixed.float2fixed_array(array_type, fractal_width, bias)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        if isFixed == True:
            f.write(str(" * array_type : {}\r\n").format(weight.dtype))
            f.write(str(" * fractal_width : {} bit\r\n").format(fractal_width))
            f.write(str(" * bit_width : {} bit\r\n").format(str(8 * np.dtype(array_type).itemsize)))
        f.write(" *\n */\n")

        # include <stdint.h>
        f.write(str("#pragma once\r\n"))
        f.write(str("#include <stdint.h>\r\n\r\n"))

        # define data_width
        if isFixed == True:
            f.write(str("#define data_width_{} {}\r\n").format(str(weight_array_name[:-2]), str(8 * np.dtype(array_type).itemsize)))
            f.write(str("#define fractal_width_{} {}\r\n\r\n").format(str(weight_array_name[:-2]), str(fractal_width)))

        # weights
        f.write(str("const uint16_t shape_{}_w[] = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\r\n" % (weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]))

        f.write("const " + type_name + " " + weight_array_name)
        f.write(str("[{}][{}][{}][{}] =\r\n").format(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]))

        write_array_ND(weight, f)
        f.write(";")
        f.write("\r\n\r\n")

        # bias
        f.write(str("const uint16_t shape_{}_b = {};\r\n").format(str(weight_array_name[:-2]), bias.shape[0]))

        f.write("const " + type_name + " " + bias_array_name)
        f.write("[%d] = " % bias.shape)

        write_array_ND(bias, f)
        f.write(";")
        f.write("\r\n")


if __name__ == "__main__":
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
            if(layers["class_name"].find("UpSampling") == -1 and layers["class_name"].find("MaxPooling2D") == -1):
                param_w = model_weights_itr[itr_counter]
                param_b = model_weights_itr[itr_counter + 1]
                itr_counter += 2
                print("weight", param_w.shape, len(param_w.shape))
                print("bias", param_b.shape, len(param_b.shape))
                write_weight_Conv2D(param_w,
                                    param_b,
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_w",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_b",
                                    "float")
                params_header_name_float.append(layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + ".h")
                write_weight_Conv2D(param_w,
                                    param_b,
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_w",
                                    layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_b",
                                    "int16_t", isFixed=True, fractal_width=8, array_type=np.int16)
                params_header_name_fix.append(layers["class_name"] + "_" + str(int(itr_counter / 2) - 1) + "_fixed16.h")
            else:
                print("This Layer has no Parameter")

    with open("keras_mnist_DCAE_params_float.h", "w") as f:
        print(params_header_name_float)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        f.write(" *\n */\n")

        for name in params_header_name_float:
            f.write('#include "' + name + '"\r\n')

    with open("keras_mnist_DCAE_params_fixed.h", "w") as f:
        print(params_header_name_fix)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        f.write(" *\n */\n")

        for name in params_header_name_fix:
            f.write('#include "' + name + '"\r\n')
