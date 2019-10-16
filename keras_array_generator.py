from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import datetime
import numpy as np
import os

fractal = 16 - 2
output_file_array_float32 = "./arrays/arrays_float32.h"
output_file_array_fix16 = "./arrays/arrays_fix16.h"
output_file_template_float32 = "template_float32.c"
output_file_template_fix16 = "template_fix16.c"

with open("keras_mnist_DCAE/keras_mnist_DCAE.json") as jfile:
    if os.path.isdir("./arrays") is False:
        os.mkdir("./arrays")

    model = load_model("keras_mnist_DCAE/keras_mnist_DCAE.h5")
    model.summary()
    model_weights_itr = model.get_weights()
    model_arrays_itr = json.load(jfile, object_pairs_hook=OrderedDict)

    arrays_fix16 = []
    arrays_float32 = []
    layer_params_fix16_dict = {"layer_name": "input_0", "depth": 1, "height": 28, "width": 28}
    layer_params_fix16 = [layer_params_fix16_dict]
    # layer_params_float32_dict = {"layer_name": "input_0", "depth": 1, "height": 28, "width": 28}
    # layers_params_float32 = [layer_params_float32_dict]

    itr_counter = {"MaxPooling2D": 0, "UpSampling2D": 0, "Conv2D": 0, "Padding2D": 0}
    array_shapes = np.array([0, 0, 0], dtype=np.uint16)  # depth height width

    for layers in model_arrays_itr["config"]["layers"]:

        layer_name = str(layers["class_name"] + "_{}").format(itr_counter[layers["class_name"]])

        if layers["class_name"].find("MaxPooling2D") != -1:
            size = (layers["config"]["pool_size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] / size[0], array_shapes[2] / size[1]], dtype=np.uint16)

            layer_params_fix16_dict = {"layer_name": layer_name,
                                       "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                       "ksize_h": size[0], "ksize_w": size[1]}

            itr_counter["MaxPooling2D"] += 1

        elif layers["class_name"].find("UpSampling2D") != -1:
            size = (layers["config"]["size"])[:]
            array_shapes = np.array([array_shapes[0], array_shapes[1] * size[0], array_shapes[2] * size[1]], dtype=np.uint16)

            layer_params_fix16_dict = {"layer_name": layer_name,
                                       "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                       "ksize_h": size[0], "ksize_w": size[1]}

            itr_counter["UpSampling2D"] += 1

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
                arrays_float32.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_height + padding[1])))
                arrays_float32.append(str("float Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_height + padding[1])))

                # write to fix16.c file
                arrays_fix16.append(str("uint16_t Padding2D_{}_depth = {}, Padding2D_{}_height = {}, Padding2D_{}_width = {};\n").format(itr_counter["Padding2D"], input_depth, itr_counter["Padding2D"], int(out_shapes_height + padding[0]), itr_counter["Padding2D"], int(out_shapes_height + padding[1])))
                arrays_fix16.append(str("int16_t Padding2D_{}_array[{}][{}][{}];\n\n").format(itr_counter["Padding2D"], input_depth, int(out_shapes_height + padding[0]), int(out_shapes_height + padding[1])))

                layer_params_fix16_dict = {"layer_name": str("Padding2D_{}").format(itr_counter["Padding2D"]),
                                           "depth": input_depth, "height": int(out_shapes_height + padding[0]), "width": int(out_shapes_width + padding[1]),
                                           "padding_h": int(padding[0] / 2), "padding_w": int(padding[1] / 2)}
                layer_params_fix16.append(layer_params_fix16_dict)

                itr_counter["Padding2D"] += 1

            else:
                out_shapes_height = (input_shapes[0] - kernel_shapes[0]) / strides[0] + 1
                out_shapes_width = (input_shapes[1] - kernel_shapes[1]) / strides[1] + 1

            # depth first, not last
            array_shapes = np.array([layers["config"]["filters"], out_shapes_height, out_shapes_width], dtype=np.uint16)

            layer_params_fix16_dict = {"layer_name": layer_name,
                                       "depth": array_shapes[0], "height": array_shapes[1], "width": array_shapes[2],
                                       "ksize_h": kernel_shapes[0], "ksize_w": kernel_shapes[1],
                                       "bias_length": array_shapes[0], "activation": layers["config"]["activation"], "fractal": fractal}

            itr_counter["Conv2D"] += 1

        else:
            print("This Layer is not available OTL")

        # write to float32.c file
        arrays_float32.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_float32.append(str("float " + layer_name + "_array[{}][{}][{}];\n\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

        # write to fix16.c file
        arrays_fix16.append(str("uint16_t " + layer_name + "_depth = {}, " + layer_name + "_height = {}, " + layer_name + "_width = {};\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))
        arrays_fix16.append(str("int16_t " + layer_name + "_array[{}][{}][{}];\n\n").format(array_shapes[0], array_shapes[1], array_shapes[2]))

        layer_params_fix16.append(layer_params_fix16_dict)

    with open(output_file_array_float32, "w") as f:
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

    with open(output_file_array_fix16, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        for i in arrays_fix16:
            f.write(i)

    with open(output_file_template_fix16, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdio.h>\n\n")

        f.write('#include "array_printf_fix16.h"\n')
        f.write('#include "arrays/arrays_fix16.h"\n')
        f.write('#include "layers/layers.h"\n')
        f.write('#include "weights/weights_fix16.h"\n\n')

        f.write("int main(void){\n")
        for i in layer_params_fix16:
            print(i)
            if i["layer_name"].find("input") != -1:
                f.write(str("\tuint16_t " + i["layer_name"] + "_depth = {}, " + i["layer_name"] + "_height = {}, " + i["layer_name"] + "_width = {};\n").format(i["depth"], i["height"], i["width"]))
                f.write(str("\tint16_t " + i["layer_name"] + "_array[{}][{}][{}];\n\n").format(i["depth"], i["height"], i["width"]))
            elif i["layer_name"].find("Padding2D") != -1:
                f.write(str("\tpadding2d_fix16({}, {},\n\t").format(i["padding_h"], i["padding_w"]))
                f.write(str("{0}_depth ,{0}_height ,{0}_width ,{0}_array,\n\t").format(i_old["layer_name"]))
                f.write(str("{0}_height ,{0}_width ,{0}_array);\n").format(i["layer_name"]))

            i_old = i.copy()
        f.write("\treturn(0);\n")
        f.write("}\n")
