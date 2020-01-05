from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
from keras_mnist_DCAE import float2fixed
import os


fractal = 16 - 3


def write_array_1D(array, f):
    f.write("{")
    for length in range(array.shape[0]):
        if(str(array.dtype).find("float") == -1):
            f.write(str("{:5d}").format(int(array[length])))
        else:
            array = array.astype(np.float32)
            # f.write(str("{0:.20f}").format(array[length]))
            f.write(str("{:e}").format(array[length]))
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


def write_weight_SeparableConv2D_c(weight_depthwise, weight_pointwise, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed=False, fractal_width=0, array_type=None):
    # Weight generated by Keras has 4D array with (height, width, input_depth, output_depth)
    # Transpose weight axis from (height, width, input_depth, output_depth) to (output_depth, input_depth ,height, width)

    with open(file_name, 'w') as f:

        # reshape weight array
        weight_depthwise = weight_depthwise.transpose(3, 2, 0, 1)
        weight_pointwise = weight_pointwise.transpose(3, 2, 0, 1)
        bias_pointwise = np.copy(bias)

        if isFixed is True:
            weight_depthwise = float2fixed.float2fixed_array(array_type, fractal_width, weight_depthwise)
            weight_pointwise = float2fixed.float2fixed_array(array_type, fractal_width, weight_pointwise)
            bias_pointwise = float2fixed.float2fixed_array(array_type, fractal_width, bias_pointwise)

        # bias_depthwise = np.zeros((bias.shape), dtype=bias_pointwise.dtype)
        bias_depthwise = np.zeros((weight_pointwise.shape[1],), dtype=bias_pointwise.dtype)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        if isFixed is True:
            f.write(str(" * array_type : {}\n").format(weight_depthwise.dtype))
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

        # weights(depthwise)
        f.write(str("const uint16_t shape_{}_w_d[] = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight_depthwise.shape[0], weight_depthwise.shape[1], weight_depthwise.shape[2], weight_depthwise.shape[3]))

        f.write("const " + type_name + " " + weight_array_name + "_d")
        f.write(str("[{}][{}][{}][{}] =\n").format(weight_depthwise.shape[0], weight_depthwise.shape[1], weight_depthwise.shape[2], weight_depthwise.shape[3]))

        write_array_ND(weight_depthwise, f)
        f.write(";")
        f.write("\n\n")

        # weights(pointwise)
        f.write(str("const uint16_t shape_{}_w_p[] = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight_pointwise.shape[0], weight_pointwise.shape[1], weight_pointwise.shape[2], weight_pointwise.shape[3]))

        f.write("const " + type_name + " " + weight_array_name + "_p")
        # f.write(str("[{}][{}][{}][{}] =\n").format(weight_pointwise.shape[0], weight_pointwise.shape[1], weight_pointwise.shape[2], weight_pointwise.shape[3]))
        f.write(str("[{}] =\n").format(weight_pointwise.shape[0] * weight_pointwise.shape[1] * weight_pointwise.shape[2] * weight_pointwise.shape[3]))

        write_array_ND(weight_pointwise.reshape(-1), f)
        f.write(";")
        f.write("\n\n")

        # bias(depthwise)
        f.write(str("const uint16_t shape_{}_b_d = {};\n").format(str(weight_array_name[:-2]), bias_depthwise.shape[0]))

        f.write("const " + type_name + " " + bias_array_name + "_d")
        f.write("[%d] = " % bias_depthwise.shape)

        write_array_ND(bias_depthwise, f)
        f.write(";")
        f.write("\n")

        # bias(pointwise)
        f.write(str("const uint16_t shape_{}_b_p = {};\n").format(str(weight_array_name[:-2]), bias.shape[0]))

        f.write("const " + type_name + " " + bias_array_name + "_p")
        f.write("[%d] = " % bias_pointwise.shape)

        write_array_ND(bias_pointwise, f)
        f.write(";")
        f.write("\n")


def write_weight_SeparableConv2D_cpp(weight_depthwise, weight_pointwise, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed=False, fractal_width=0, array_type=None):
    # Weight generated by Keras has 4D array with (height, width, input_depth, output_depth)
    # Transpose weight axis from (height, width, input_depth, output_depth) to (output_depth, input_depth ,height, width)

    with open(file_name, 'w') as f:

        # reshape weight array
        weight_depthwise = weight_depthwise.transpose(3, 2, 0, 1)
        weight_pointwise = weight_pointwise.transpose(3, 2, 0, 1)
        bias_pointwise = np.copy(bias)

        if isFixed is True:
            weight_depthwise = float2fixed.float2fixed_array(array_type, fractal_width, weight_depthwise)
            weight_pointwise = float2fixed.float2fixed_array(array_type, fractal_width, weight_pointwise)
            bias_pointwise = float2fixed.float2fixed_array(array_type, fractal_width, bias)

        bias_depthwise = np.zeros((weight_pointwise.shape[1],), dtype=bias_pointwise.dtype)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        if isFixed is True:
            f.write(str(" * array_type : {}\n").format(weight_depthwise.dtype))
            f.write(str(" * fractal_width : {} bit\n").format(fractal_width))
            f.write(str(" * bit_width : {} bit\n").format(str(8 * np.dtype(array_type).itemsize)))
        f.write(" *\n */\n")

        # include <cstdint> and <vector>
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")
        f.write("using namespace std;\n\n")

        # define data_width
        if isFixed is True:
            f.write(str("#define data_width_{} {}\n").format(str(weight_array_name[:-2]), str(8 * np.dtype(array_type).itemsize)))
            f.write(str("#define fractal_width_{} {}\n\n").format(str(weight_array_name[:-2]), str(fractal_width)))

        # weights(depthwise)
        f.write(str("const vector< uint16_t> shape_{}_w_d = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight_depthwise.shape[0], weight_depthwise.shape[1], weight_depthwise.shape[2], weight_depthwise.shape[3]))

        f.write("const vector< vector< vector< vector< " + type_name + "> > > > " + weight_array_name + "_d =\n")

        write_array_ND(weight_depthwise, f)
        f.write(";")
        f.write("\n\n")

        # weights(pointwise)
        f.write(str("const vector< uint16_t> shape_{}_w_p = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight_pointwise.shape[0], weight_pointwise.shape[1], weight_pointwise.shape[2], weight_pointwise.shape[3]))

        f.write("const vector< vector< vector< vector< " + type_name + "> > > > " + weight_array_name + "_p =\n")

        write_array_ND(weight_pointwise, f)
        f.write(";")
        f.write("\n\n")

        # bias(depthwise)
        f.write(str("const uint16_t shape_{}_b_d = {};\n").format(str(weight_array_name[:-2]), bias_depthwise.shape[0]))

        f.write("const vector< " + type_name + "> " + bias_array_name + "_d = ")

        write_array_ND(bias_depthwise, f)
        f.write(";")
        f.write("\n")

        # bias(pointwise)
        f.write(str("const uint16_t shape_{}_b_p = {};\n").format(str(weight_array_name[:-2]), bias_pointwise.shape[0]))

        f.write("const vector< " + type_name + "> " + bias_array_name + "_p = ")

        write_array_ND(bias_pointwise, f)
        f.write(";")
        f.write("\n")


def write_weight_Conv2D_c(weight, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed=False, fractal_width=0, array_type=None):
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
        f.write("{%d, %d, %d, %d};\n" % (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]))

        f.write("const " + type_name + " " + weight_array_name)
        f.write(str("[{}][{}][{}][{}] =\n").format(weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]))

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


def write_weight_Conv2D_cpp(weight, bias, file_name, weight_array_name, bias_array_name, type_name, isFixed=False, fractal_width=0, array_type=None):
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

        # include <cstdint> and <vector>
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")
        f.write("using namespace std;\n\n")

        # define data_width
        if isFixed is True:
            f.write(str("#define data_width_{} {}\n").format(str(weight_array_name[:-2]), str(8 * np.dtype(array_type).itemsize)))
            f.write(str("#define fractal_width_{} {}\n\n").format(str(weight_array_name[:-2]), str(fractal_width)))

        # weights
        f.write(str("const vector< uint16_t> shape_{}_w = ").format(str(weight_array_name[:-2])))
        f.write("{%d, %d, %d, %d};\n" % (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]))

        f.write("const vector< vector< vector< vector< " + type_name + "> > > > " + weight_array_name + " =\n")

        write_array_ND(weight, f)
        f.write(";")
        f.write("\n\n")

        # bias
        f.write(str("const uint16_t shape_{}_b = {};\n").format(str(weight_array_name[:-2]), bias.shape[0]))

        f.write("const vector< " + type_name + "> " + bias_array_name + " = ")
        # f.write(" = " % bias.shape)

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
        Conv2D_counter = 0
        SepConv2D_counter = 0
        for layers in model_layers_itr["config"]["layers"]:
            print()
            print(layers["class_name"])
            if(layers["class_name"].find("SeparableConv2D") != -1):
                print(layers["config"]["batch_input_shape"])
                print(layers["config"]["filters"])
                param_w_d = model_weights_itr[itr_counter]
                param_w_p = model_weights_itr[itr_counter + 1]
                param_b = model_weights_itr[itr_counter + 2]
                itr_counter += 3
                print("weight_depthwise", param_w_d.shape, len(param_w_d.shape), param_w_d.dtype)
                print("weight_pointwise", param_w_p.shape, len(param_w_p.shape), param_w_p.dtype)
                print("bias", param_b.shape, len(param_b.shape), param_b.dtype)

                write_weight_SeparableConv2D_c(param_w_d,
                                               param_w_p,
                                               param_b,
                                               "weights_c/" + layers["class_name"] + "_" + str(SepConv2D_counter) + "_float32.h",
                                               layers["class_name"] + "_" + str(SepConv2D_counter) + "_w",
                                               layers["class_name"] + "_" + str(SepConv2D_counter) + "_b",
                                               "float")

                params_header_name_float.append(layers["class_name"] + "_" + str(SepConv2D_counter) + "_float32.h")

                write_weight_SeparableConv2D_c(param_w_d,
                                               param_w_p,
                                               param_b,
                                               "weights_c/" + layers["class_name"] + "_" + str(SepConv2D_counter) + "_fix16.h",
                                               layers["class_name"] + "_" + str(SepConv2D_counter) + "_w",
                                               layers["class_name"] + "_" + str(SepConv2D_counter) + "_b",
                                               "int16_t", isFixed=True, fractal_width=fractal, array_type=np.int16)

                params_header_name_fix.append(layers["class_name"] + "_" + str(SepConv2D_counter) + "_fix16.h")

                write_weight_SeparableConv2D_cpp(param_w_d,
                                                 param_w_p,
                                                 param_b,
                                                 "weights_cpp/" + layers["class_name"] + "_" + str(SepConv2D_counter) + "_fix16.h",
                                                 layers["class_name"] + "_" + str(SepConv2D_counter) + "_w",
                                                 layers["class_name"] + "_" + str(SepConv2D_counter) + "_b",
                                                 "int16_t", isFixed=True, fractal_width=fractal, array_type=np.int16)

                write_weight_SeparableConv2D_cpp(param_w_d,
                                                 param_w_p,
                                                 param_b,
                                                 "weights_cpp/" + layers["class_name"] + "_" + str(SepConv2D_counter) + "_float32.h",
                                                 layers["class_name"] + "_" + str(SepConv2D_counter) + "_w",
                                                 layers["class_name"] + "_" + str(SepConv2D_counter) + "_b",
                                                 "float")
                SepConv2D_counter += 1

            elif(layers["class_name"].find("Conv2D") != -1):
                print(layers["config"]["batch_input_shape"])
                print(layers["config"]["filters"])
                param_w = model_weights_itr[itr_counter]
                param_b = model_weights_itr[itr_counter + 1]
                itr_counter += 2
                print("weight", param_w.shape, len(param_w.shape), param_w.dtype)
                print("bias", param_b.shape, len(param_b.shape), param_b.dtype)

                write_weight_Conv2D_c(param_w,
                                      param_b,
                                      "weights_c/" + layers["class_name"] + "_" + str(Conv2D_counter) + "_float32.h",
                                      layers["class_name"] + "_" + str(Conv2D_counter) + "_w",
                                      layers["class_name"] + "_" + str(Conv2D_counter) + "_b",
                                      "float")

                params_header_name_float.append(layers["class_name"] + "_" + str(Conv2D_counter) + "_float32.h")

                write_weight_Conv2D_c(param_w,
                                      param_b,
                                      "weights_c/" + layers["class_name"] + "_" + str(Conv2D_counter) + "_fix16.h",
                                      layers["class_name"] + "_" + str(Conv2D_counter) + "_w",
                                      layers["class_name"] + "_" + str(Conv2D_counter) + "_b",
                                      "int16_t", isFixed=True, fractal_width=fractal, array_type=np.int16)

                params_header_name_fix.append(layers["class_name"] + "_" + str(Conv2D_counter) + "_fix16.h")

                write_weight_Conv2D_cpp(param_w,
                                        param_b,
                                        "weights_cpp/" + layers["class_name"] + "_" + str(Conv2D_counter) + "_fix16.h",
                                        layers["class_name"] + "_" + str(Conv2D_counter) + "_w",
                                        layers["class_name"] + "_" + str(Conv2D_counter) + "_b",
                                        "int16_t", isFixed=True, fractal_width=fractal, array_type=np.int16)

                write_weight_Conv2D_cpp(param_w,
                                        param_b,
                                        "weights_cpp/" + layers["class_name"] + "_" + str(Conv2D_counter) + "_float32.h",
                                        layers["class_name"] + "_" + str(Conv2D_counter) + "_w",
                                        layers["class_name"] + "_" + str(Conv2D_counter) + "_b",
                                        "float")

                Conv2D_counter += 1
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

        f.write("\n#define fractal_width_input_0 {}\n".format(fractal))

    with open("./weights_cpp/weights_float32.h", "w") as f:
        # print(params_header_name_float)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n */\n")

        for name in params_header_name_float:
            f.write('#include "' + name + '"\n')

    with open("./weights_cpp/weights_fix16.h", "w") as f:
        # print(params_header_name_fix)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n */\n")

        for name in params_header_name_fix:
            f.write('#include "' + name + '"\n')

        f.write("\n#define fractal_width_input_0 {}\n".format(fractal))