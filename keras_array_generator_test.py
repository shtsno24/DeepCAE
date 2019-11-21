from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
import os

output_file_C_array_float32 = "./arrays_c/arrays_float32.h"
output_file_C_array_fix16 = "./arrays_c/arrays_fix16.h"
output_file_C_template_float32 = "./template_c/template_float32.c"
output_file_C_template_fix16 = "./template_c/template_fix16.c"

output_file_Cpp_array_float32 = "./arrays_cpp/arrays_float32.h"
output_file_Cpp_array_fix16 = "./arrays_cpp/arrays_fix16.h"
output_file_Cpp_template_float32 = "./template_cpp/template_float32.cpp"
output_file_Cpp_template_fix16 = "./template_cpp/template_fix16.cpp"

output_languages = ["c", "cpp"]
output_files = ["template", "arrays"]
output_precision = ["fix16", "float32"]

with open("keras_mnist_DCAE/keras_mnist_DCAE.json") as jfile:

    for files in output_files:
        for lang in output_languages:
            if os.path.isdir(str("./{}_{}").format(files, lang)) is False:
                os.mkdir(str("./{}_{}").format(files, lang))

    model = load_model("keras_mnist_DCAE/keras_mnist_DCAE.h5")
    model.summary()
    model_weights_itr = model.get_weights()
    model_arrays_itr = json.load(jfile, object_pairs_hook=OrderedDict)

    arrays_fix16 = []
    arrays_float32 = []
    layer_params_dict = {"layer_name": "input_0", "depth": 1, "height": 28, "width": 28}
    layer_params = [layer_params_dict]

    itr_counter = {"MaxPooling2D": 0, "UpSampling2D": 0, "Conv2D": 0, "Padding2D": 0, "SeparableConv2D": 0, "DepthwiseConv2D": 0, "PointwiseConv2D": 0, "input_0": 1}
    array_shapes = np.array([0, 0, 0], dtype=np.uint16)  # depth height width

    for layers in model_arrays_itr["config"]["layers"]:

        layer_name = str(layers["class_name"] + "_{}").format(itr_counter[layers["class_name"]])

        if layers["class_name"].find("MaxPooling2D") != -1:
            size = (layers["config"]["pool_size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] / size[0], array_shapes[2] / size[1]], dtype=np.uint16)

            layer_params_dict = {"layer_name": layer_name,
                                 "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                 "ksize_h": size[0], "ksize_w": size[1]}
            layer_params.append(layer_params_dict)
            itr_counter["MaxPooling2D"] += 1

        elif layers["class_name"].find("UpSampling2D") != -1:
            size = (layers["config"]["size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] * size[0], array_shapes[2] * size[1]], dtype=np.uint16)

            layer_params_dict = {"layer_name": layer_name,
                                 "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                 "ksize_h": size[0], "ksize_w": size[1]}
            layer_params.append(layer_params_dict)
            itr_counter["UpSampling2D"] += 1

        elif layers["class_name"].find("SeparableConv2D") != -1:
            if itr_counter["SeparableConv2D"] == 0:
                input_shapes = (layers["config"]["batch_input_shape"])[1:3]
                input_depth = (layers["config"]["batch_input_shape"])[3]
            else:
                input_shapes = array_shapes[1:3]
                input_depth = array_shapes[0]
            kernel_shapes = (layers["config"]["kernel_size"])[:]
            strides = (layers["config"]["strides"])[:]
            if layers["config"]["padding"] == "same":
                out_shapes_height = input_shapes[0] / strides[0]
                out_shapes_width = input_shapes[1] / strides[0]

                # calc padding_length(not half length)
                padding = np.array([np.max(kernel_shapes[0] - strides[0], 0), np.max(kernel_shapes[1] - strides[1], 0)])

                # generate padding layers
                # write to float32.c file
                # arrays_float32.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_width + padding[1])))
                # arrays_float32.append(str("float Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_width + padding[1])))

                # write to fix16.c file
                # arrays_fix16.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_width + padding[1])))
                # arrays_fix16.append(str("int16_t Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_width + padding[1])))

                layer_params_dict = {"layer_name": str("Padding2D_{}").format(itr_counter["Padding2D"]),
                                     "depth": input_depth, "height": int(out_shapes_height + padding[0]), "width": int(out_shapes_width + padding[1]),
                                     "padding_h": int(padding[0] / 2), "padding_w": int(padding[1] / 2)}
                layer_params.append(layer_params_dict)
                itr_counter["Padding2D"] += 1

            else:
                out_shapes_height = (input_shapes[0] - kernel_shapes[0]) / strides[0] + 1
                out_shapes_width = (input_shapes[1] - kernel_shapes[1]) / strides[1] + 1

            # write middle array to float32.c file
            # arrays_float32.append(str("uint16_t SeparableConv2D_{}_m_depth = {}, SeparableConv2D_{}_m_height = {}, SeparableConv2D_{}_m_width = {};\n").format(itr_counter["SeparableConv2D"], input_depth, itr_counter["SeparableConv2D"], int(out_shapes_height), itr_counter["SeparableConv2D"], int(out_shapes_width)))
            # arrays_float32.append(str("float SeparableConv2D_{}_m_array[{}][{}][{}];\n\n").format(itr_counter["SeparableConv2D"], input_depth, int(out_shapes_height), int(out_shapes_width)))

            # write middle array to fix16.c file
            # arrays_fix16.append(str("uint16_t SeparableConv2D_{}_m_depth = {}, SeparableConv2D_{}_m_height = {}, SeparableConv2D_{}_m_width = {};\n").format(itr_counter["SeparableConv2D"], input_depth, itr_counter["SeparableConv2D"], int(out_shapes_height), itr_counter["SeparableConv2D"], int(out_shapes_width)))
            # arrays_fix16.append(str("int16_t SeparableConv2D_{}_m_array[{}][{}][{}];\n\n").format(itr_counter["SeparableConv2D"], input_depth, int(out_shapes_height), int(out_shapes_width)))

            # depth first, not last
            array_shapes = np.array([layers["config"]["filters"], out_shapes_height, out_shapes_width], dtype=np.uint16)

            layer_params_dict = {"layer_name": str("DepthwiseConv2D_{}").format(itr_counter["DepthwiseConv2D"]),
                                 "depth": input_depth, "height": array_shapes[1], "width": array_shapes[2],  # output shape
                                 "ksize_h": kernel_shapes[0], "ksize_w": kernel_shapes[1],
                                 "bias_length": input_depth, "activation": layers["config"]["activation"]}
            layer_params.append(layer_params_dict)
            layer_params_dict = {"layer_name": str("PointwiseConv2D_{}").format(itr_counter["PointwiseConv2D"]),
                                 "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],  # output shape
                                 "ksize_h": 1, "ksize_w": 1,
                                 "bias_length": array_shapes[0], "activation": layers["config"]["activation"]}
            layer_params.append(layer_params_dict)
            layer_params_dict = {"layer_name": layer_name,
                                 "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],  # output shape
                                 "ksize_h": kernel_shapes[0], "ksize_w": kernel_shapes[1],
                                 "bias_length": array_shapes[0], "activation": layers["config"]["activation"]}
            layer_params.append(layer_params_dict)
            itr_counter["SeparableConv2D"] += 1
            itr_counter["DepthwiseConv2D"] += 1
            itr_counter["PointwiseConv2D"] += 1

        elif layers["class_name"].find("Conv2D") != -1:
            if itr_counter["Conv2D"] == 0:
                input_shapes = (layers["config"]["batch_input_shape"])[1:3]
                input_depth = (layers["config"]["batch_input_shape"])[3]
            else:
                input_shapes = array_shapes[1:3]
                input_depth = array_shapes[0]
            kernel_shapes = (layers["config"]["kernel_size"])[:]
            strides = (layers["config"]["strides"])[:]
            if layers["config"]["padding"] == "same":
                out_shapes_height = input_shapes[0] / strides[0]
                out_shapes_width = input_shapes[1] / strides[0]

                # calc padding_length(not half length)
                padding = np.array([np.max(kernel_shapes[0] - strides[0], 0), np.max(kernel_shapes[1] - strides[1], 0)])

                # generate padding layers
                # write to float32.c file
                # arrays_float32.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_width + padding[1])))
                # arrays_float32.append(str("float Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_width + padding[1])))

                # write to fix16.c file
                # arrays_fix16.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_width + padding[1])))
                # arrays_fix16.append(str("int16_t Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_width + padding[1])))

                layer_params_dict = {"layer_name": str("Padding2D_{}").format(itr_counter["Padding2D"]),
                                     "depth": input_depth, "height": int(out_shapes_height + padding[0]), "width": int(out_shapes_width + padding[1]),
                                     "padding_h": int(padding[0] / 2), "padding_w": int(padding[1] / 2)}
                layer_params.append(layer_params_dict)
                itr_counter["Padding2D"] += 1

            else:
                out_shapes_height = (input_shapes[0] - kernel_shapes[0]) / strides[0] + 1
                out_shapes_width = (input_shapes[1] - kernel_shapes[1]) / strides[1] + 1

            # depth first, not last
            array_shapes = np.array([layers["config"]["filters"], out_shapes_height, out_shapes_width], dtype=np.uint16)

            layer_params_dict = {"layer_name": layer_name,
                                 "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                 "ksize_h": kernel_shapes[0], "ksize_w": kernel_shapes[1],
                                 "bias_length": array_shapes[0], "activation": layers["config"]["activation"]}
            layer_params.append(layer_params_dict)
            itr_counter["Conv2D"] += 1

        else:
            print("This Layer is not available OTL")

        # write to float32.c file
        # arrays_float32.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        # arrays_float32.append(str("float " + layer_name + "_array[{}][{}][{}];\n\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

        # write to fix16.c file
        # arrays_fix16.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        # arrays_fix16.append(str("int16_t " + layer_name + "_array[{}][{}][{}];\n\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

    for langs in output_languages:
        for files in output_files:
            for precise in output_precision:
                # "./arrays_C/arrays_float32.h"
                if files == "arrays":
                    buff_name = str("./{0}_{1}/{0}_{2}.h").format(files, langs, precise)
                else:
                    buff_name = str("./{0}_{1}/{0}_{2}.{1}").format(files, langs, precise)
                print(buff_name)

    for layer_p in layer_params:
        print(layer_p["layer_name"], layer_p["depth"], layer_p["height"], layer_p["width"], end=" ")
        if(layer_p["layer_name"].find("Conv2D") != -1):
            print(layer_p["ksize_h"], layer_p["ksize_w"], layer_p["bias_length"])
        else:
            print()

        if(layer_p["layer_name"] == "input_0"):
            max_array_size = layer_p["depth"] * layer_p["height"] * layer_p["width"]
        if(layer_p["depth"] * layer_p["height"] * layer_p["width"] > max_array_size):
            max_array_size = layer_p["depth"] * layer_p["height"] * layer_p["width"]

    print(max_array_size)
    print(itr_counter)


"""
    with open(output_file_C_array_float32, "w") as f:
        todaytime = str(datetime.datetime.today())
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        for i in arrays_float32:
            f.write(i)

    with open(output_file_C_array_fix16, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        for i in arrays_fix16:
            f.write(i)

    with open(output_file_C_template_fix16, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdio.h>\n\n")

        f.write('#include "test_data/test_data.h"\n')
        f.write('#include "layers_c/array_printf_fix16.h"\n')
        f.write('#include "arrays_c/arrays_fix16.h"\n')
        f.write('#include "layers_c/layers.h"\n')
        f.write('#include "weights_c/weights_fix16.h"\n\n')

        for i in layer_params:
            print(i)
            if i["layer_name"].find("input") != -1:
                f.write(str("int network(int16_t input_data[{0}*{1}*{2}], int16_t output_data[{0}*{1}*{2}]){{\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tuint16_t " + i["layer_name"] + "_depth = {}, " + i["layer_name"] + "_height = {}, " + i["layer_name"] + "_width = {};\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tint16_t " + i["layer_name"] + "_array[{}][{}][{}];\n\n").format(i["depth"], i["height"], i["width"]))

                f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(i["layer_name"]))
                f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(i["layer_name"]))
                f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(i["layer_name"]))
                f.write("\t\t\t\t{0}_array[depth][height][width] = input_data[depth * {0}_height * {0}_width + height * {0}_width + width];\n".format(i["layer_name"]))
                f.write("\t\t\t}\n")
                f.write("\t\t}\n")
                f.write("\t}\n")


                f.write('\tFILE* fp = fopen("template_input_fix16.tsv", "w");\n\t')
                f.write("array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\\t', fp, fractal_width_input_0);\n\t")
                f.write("fclose(fp);\n\n")

            elif i["layer_name"].find("Padding2D") != -1:
                f.write(str("\tpadding2d_fix16({}, {},\n\t").format(i["padding_h"], i["padding_w"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_height, {0}_width, (int16_t*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("MaxPooling2D") != -1:
                f.write(str("\tmax_pooling2d_fix16({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("UpSampling2D") != -1:
                f.write(str("\tup_sampling2d_fix16({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("DepthwiseConv2D") != -1:
                pass

            elif i["layer_name"].find("SeparableConv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tseparable_conv2d_fix16({0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array, (int16_t*) {0}_m_array,\n\t").format(i["layer_name"]))
                f.write(str("(int16_t*) {0}_b_d, (int16_t*) {0}_b_p,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, (int16_t*) {2}_w_d, (int16_t*) {2}_w_p, {3}, fractal_width_{2});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            elif i["layer_name"].find("Conv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tconv2d_fix16({0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("(int16_t*) {}_b,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, (int16_t*) {2}_w, {3}, fractal_width_{2});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            i_old = i.copy()

        f.write('\tfp = fopen("template_output_fix16.tsv", "w");\n\t')
        f.write(str("array_fprintf_2D_fix16({0}_height, {0}_width, {0}_array[0], '\\t', fp, fractal_width_{0});\n\t").format(i_old["layer_name"]))
        f.write("fclose(fp);\n\n")

        f.write("\treturn(0);\n")
        f.write("}\n")

    with open(output_file_C_template_float32, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdio.h>\n\n")

        f.write('#include "test_data/test_data.h"\n')
        f.write('#include "layers_c/array_printf_float32.h"\n')
        f.write('#include "arrays_c/arrays_float32.h"\n')
        f.write('#include "layers_c/layers.h"\n')
        f.write('#include "weights_c/weights_float32.h"\n\n')

        for i in layer_params:
            print(i)
            if i["layer_name"].find("input") != -1:
                f.write(str("int network(float input_data[{0}*{1}*{2}], float output_data[{0}*{1}*{2}]){{\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tuint16_t " + i["layer_name"] + "_depth = {}, " + i["layer_name"] + "_height = {}, " + i["layer_name"] + "_width = {};\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tfloat " + i["layer_name"] + "_array[{}][{}][{}];\n\n").format(i["depth"], i["height"], i["width"]))

                f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(i["layer_name"]))
                f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(i["layer_name"]))
                f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(i["layer_name"]))
                f.write("\t\t\t\t{0}_array[depth][height][width] = input_data[depth * {0}_height * {0}_width + height * {0}_width + width];\n".format(i["layer_name"]))
                f.write("\t\t\t}\n")
                f.write("\t\t}\n")
                f.write("\t}\n")

                f.write('\tFILE* fp = fopen("template_input_float32.tsv", "w");\n\t')
                f.write("array_fprintf_2D_float32(input_0_height, input_0_width, input_0_array[0], '\\t', fp);\n\t")
                f.write("fclose(fp);\n\n")

            elif i["layer_name"].find("Padding2D") != -1:
                f.write(str("\tpadding2d_float32({}, {},\n\t").format(i["padding_h"], i["padding_w"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_height, {0}_width, (float*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("MaxPooling2D") != -1:
                f.write(str("\tmax_pooling2d_float32({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("UpSampling2D") != -1:
                f.write(str("\tup_sampling2d_float32({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("SeparableConv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tseparable_conv2d_float32({0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array, (float*) {0}_m_array,\n\t").format(i["layer_name"]))
                f.write(str("(float*) {0}_b_d, (float*) {0}_b_p,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, (float*) {2}_w_d, (float*) {2}_w_p, {3});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            elif i["layer_name"].find("Conv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tconv2d_float32({0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, (float*) {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("(float*) {}_b,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, (float*) {2}_w, {3});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            i_old = i.copy()

        f.write('\tfp = fopen("template_output_float32.tsv", "w");\n\t')
        f.write(str("array_fprintf_2D_float32({0}_height, {0}_width, {0}_array[0], '\\t', fp);\n\t").format(i_old["layer_name"]))
        f.write("fclose(fp);\n\n")

        f.write("\treturn(0);\n")
        f.write("}\n")

    with open("./template_cpp/template_fix16.cpp", "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")

        f.write('#include "./../test_data/test_data.h"\n')
        f.write('#include "./../layers_cpp/array_printf_fix16.h"\n')
        f.write('#include "./../arrays_cpp/arrays_fix16.h"\n')
        f.write('#include "./../layers_cpp/layers.h"\n')
        f.write('#include "./../weights_cpp/weights_fix16.h"\n\n')
        f.write('using namespace std;\n\n')

        for i in layer_params:
            print(i)
            if i["layer_name"].find("input") != -1:
                f.write(str("int network(int16_t input_data[{0}*{1}*{2}], int16_t output_data[{0}*{1}*{2}]){{\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tuint16_t " + i["layer_name"] + "_depth = {}, " + i["layer_name"] + "_height = {}, " + i["layer_name"] + "_width = {};\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tvector< vector< vector< int16_t> > > {0}_array({0}_depth, vector< vector < int16_t> >({0}_height, vector< int16_t>({0}_width)));\n\n").format(i["layer_name"]))

                f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(i["layer_name"]))
                f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(i["layer_name"]))
                f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(i["layer_name"]))
                f.write("\t\t\t\t{0}_array[depth][height][width] = input_data[depth * {0}_height * {0}_width + height * {0}_width + width];\n".format(i["layer_name"]))
                f.write("\t\t\t}\n")
                f.write("\t\t}\n")
                f.write("\t}\n")

                f.write('\tofstream fp("template_input_fix16.tsv");\n\t')
                f.write("array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\\t', fp, fractal_width_input_0);\n\t")
                f.write("fp.close();\n\n")

            elif i["layer_name"].find("Padding2D") != -1:
                f.write(str("\tpadding2d_fix16({}, {},\n\t").format(i["padding_h"], i["padding_w"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("MaxPooling2D") != -1:
                f.write(str("\tmax_pooling2d_fix16({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("UpSampling2D") != -1:
                f.write(str("\tup_sampling2d_fix16({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("SeparableConv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tseparable_conv2d_fix16({0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("{0}_b_d, {0}_b_p,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, {2}_w_d, {2}_w_p, {3}, fractal_width_{2});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            elif i["layer_name"].find("Conv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tconv2d_fix16({0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("{}_b,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, {2}_w, {3}, fractal_width_{2});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            i_old = i.copy()

        f.write('\tfp.open("template_output_fix16.tsv");\n\t')
        f.write(str("array_fprintf_2D_fix16({0}_height, {0}_width, {0}_array[0], '\\t', fp, fractal_width_{0});\n\t").format(i_old["layer_name"]))
        f.write("fp.close();\n\n")

        f.write("\treturn(0);\n")
        f.write("}\n")

    with open("./arrays_cpp/arrays_fix16.h", "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")
        f.write("using namespace std;\n\n")

        for i in layer_params:
            f.write(str("uint16_t {0}_depth = {1}, {0}_height = {2}, {0}_width = {3};\n").format(i["layer_name"], i["depth"], i["height"], i["width"]))
            f.write(str("vector< vector< vector< int16_t> > > {0}_array({0}_depth, vector< vector < int16_t> >({0}_height, vector< int16_t>({0}_width)));\n\n").format(i["layer_name"]))

    with open("./template_cpp/template_float32.cpp", "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")

        f.write('#include "./../test_data/test_data.h"\n')
        f.write('#include "./../layers_cpp/array_printf_float32.h"\n')
        f.write('#include "./../arrays_cpp/arrays_float32.h"\n')
        f.write('#include "./../layers_cpp/layers.h"\n')
        f.write('#include "./../weights_cpp/weights_float32.h"\n\n')
        f.write('using namespace std;\n\n')

        for i in layer_params:
            print(i)
            if i["layer_name"].find("input") != -1:
                f.write(str("int network(float input_data[{0}*{1}*{2}], float output_data[{0}*{1}*{2}]){{\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tuint16_t " + i["layer_name"] + "_depth = {}, " + i["layer_name"] + "_height = {}, " + i["layer_name"] + "_width = {};\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tvector< vector< vector< float> > > {0}_array({0}_depth, vector< vector < float> >({0}_height, vector< float>({0}_width)));\n\n").format(i["layer_name"]))

                f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(i["layer_name"]))
                f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(i["layer_name"]))
                f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(i["layer_name"]))
                f.write("\t\t\t\t{0}_array[depth][height][width] = input_data[depth * {0}_height * {0}_width + height * {0}_width + width];\n".format(i["layer_name"]))
                f.write("\t\t\t}\n")
                f.write("\t\t}\n")
                f.write("\t}\n")

                f.write('\tofstream fp("template_input_float32.tsv");\n\t')
                f.write("array_fprintf_2D_float32(input_0_height, input_0_width, input_0_array[0], '\\t', fp);\n\t")
                f.write("fp.close();\n\n")

            elif i["layer_name"].find("Padding2D") != -1:
                f.write(str("\tpadding2d_float32({}, {},\n\t").format(i["padding_h"], i["padding_w"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("MaxPooling2D") != -1:
                f.write(str("\tmax_pooling2d_float32({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("UpSampling2D") != -1:
                f.write(str("\tup_sampling2d_float32({},\n\t").format(i["ksize_h"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array);\n\n").format(i["layer_name"]))

            elif i["layer_name"].find("SeparableConv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tseparable_conv2d_float32({0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("{0}_b_d, {0}_b_p,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, {2}_w_d, {2}_w_p, {3});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            elif i["layer_name"].find("Conv2D") != -1:
                if i["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                f.write(str("\tconv2d_float32({0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_depth, {0}_height, {0}_width, {0}_array,\n\t").format(i["layer_name"]))
                f.write(str("{}_b,\n\t").format(i["layer_name"]))
                f.write(str("{0}, {1}, {2}_w, {3});\n\n").format(i["ksize_h"], i["ksize_w"], i["layer_name"], relu_flag))

            i_old = i.copy()

        f.write('\tfp.open("template_output_float32.tsv");\n\t')
        f.write(str("array_fprintf_2D_float32({0}_height, {0}_width, {0}_array[0], '\\t', fp);\n\t").format(i_old["layer_name"]))
        f.write("fp.close();\n\n")

        f.write("\treturn(0);\n")
        f.write("}\n")

    with open("./arrays_cpp/arrays_float32.h", "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <vector>\n\n")
        f.write("using namespace std;\n\n")

        for i in layer_params:
            f.write(str("uint16_t {0}_depth = {1}, {0}_height = {2}, {0}_width = {3};\n").format(i["layer_name"], i["depth"], i["height"], i["width"]))
            f.write(str("vector< vector< vector< float> > > {0}_array({0}_depth, vector< vector < float> >({0}_height, vector< float>({0}_width)));\n\n").format(i["layer_name"]))
"""
