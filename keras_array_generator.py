from __future__ import print_function
import datetime
import os

from keras_model_parser import parse_keras_model

languages = ["c", "cpp"]
folders = ["arrays"]
precision = ["fix16", "float32"]


def array_writer(layer_parameters, file_name):
    if file_name.find("cpp") != -1:
        langs = "cpp"
    else:
        langs = "c"
    todaytime = str(datetime.datetime.today())

    with open(file_name, "w") as f:

        if langs == "c":
            f.write("/*\n")
            f.write(" * Author : shtsno24\n")
            f.write(" * Date : " + todaytime + "\n")
            f.write(" * Language : " + langs + "\n")
            f.write(" *\n")
            f.write(" */\n")
            f.write("#pragma once\n")
            f.write("#include <stdint.h>\n\n")
            for params in layer_parameters:
                f.write("uint16_t {0}_depth = {1}, {0}_height = {2}, {0}_width = {3};\n\n".format(params["layer_name"],
                        params["depth"],
                        params["height"],
                        params["width"]))
                if params["layer_name"] == "input_0":
                    max_array_size = params["depth"] * params["height"] * params["width"]
                if params["depth"] * params["height"] * params["width"] > max_array_size:
                    max_array_size = params["depth"] * params["height"] * params["width"]

            if file_name.find("fix16") != -1:
                f.write("\nint16_t MemBank_A[{0}], MemBank_B[{0}];\n".format(max_array_size))
            elif file_name.find("float32") != -1:
                f.write("\nfloat MemBank_A[{0}], MemBank_B[{0}];\n".format(max_array_size))

        elif langs == "cpp":
            f.write("/*\n")
            f.write(" * Author : shtsno24\n")
            f.write(" * Date : " + todaytime + "\n")
            f.write(" * Language : " + langs + "\n")
            f.write(" *\n")
            f.write(" */\n")
            f.write("#pragma once\n")
            f.write("#include <cstdint>\n")
            f.write("#include <vector>\n\n")
            f.write("using namespace std;\n\n")
            for params in layer_parameters:
                f.write("uint16_t {0}_depth = {1}, {0}_height = {2}, {0}_width = {3};\n\n".format(params["layer_name"],
                        params["depth"],
                        params["height"],
                        params["width"]))
                if params["layer_name"] == "input_0":
                    max_array_size = params["depth"] * params["height"] * params["width"]
                if params["depth"] * params["height"] * params["width"] > max_array_size:
                    max_array_size = params["depth"] * params["height"] * params["width"]

            if file_name.find("fix16") != -1:
                f.write("\nvector< int16_t> MemBank_A({0});\n".format(max_array_size))
                f.write("\nvector< int16_t> MemBank_B({0});\n".format(max_array_size))

            elif file_name.find("float32") != -1:
                f.write("\nvector< float> MemBank_A({0});\n".format(max_array_size))
                f.write("\nvector< float> MemBank_B({0});\n".format(max_array_size))


if __name__ == "__main__":

    for folder in folders:
        for lang in languages:
            if os.path.isdir(str("./{}_{}").format(folder, lang)) is False:
                os.mkdir(str("./{}_{}").format(folder, lang))

    layer_params, itr_counter = parse_keras_model("keras_mnist_DCAE/keras_mnist_DCAE.json", "keras_mnist_DCAE/keras_mnist_DCAE.h5")

    for langs in languages:
        for folder in folders:
            for precise in precision:
                # "./arrays_C/arrays_float32.h"
                buff_name = str("./{0}_{1}/{0}_{2}.h").format(folder, langs, precise)
                array_writer(layer_params, buff_name)