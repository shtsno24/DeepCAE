from __future__ import print_function
import datetime
import os

from keras_model_parser import parse_keras_model

languages = ["c", "cpp"]
folders = ["template"]
precision = ["fix16", "float32"]


def template_writer(layer_parameters, file_name):

    params = {"langs": "c", "precision": "fix16"}

    if file_name.find("cpp") != -1:
        params["langs"] = "cpp"
    else:
        params["langs"] = "c"

    if file_name.find("float") != -1:
        params["precision"] = "float"
    else:
        params["precision"] = "fix16"

    todaytime = str(datetime.datetime.today())

    with open(file_name, "w") as f:
        f.write("/*\n")
        f.write(" * author : shtsno24\n")
        f.write(" * Date : " + todaytime + "\n")
        f.write(" * Language : " + params["langs"] + "\n")
        f.write(" * Precision : " + params["precision"] + "\n")
        f.write(" *\n")
        f.write(" */\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdio.h>\n\n")

        f.write('#include "test_data/test_data.h"\n')
        f.write('#include "layers_c/array_printf_fix16.h"\n')
        f.write('#include "arrays_c/arrays_fix16.h"\n')
        f.write('#include "layers_c/layers.h"\n')
        f.write('#include "weights_c/weights_fix16.h"\n\n')

        Memory_Bank = "MemBank_A"
        Memory_Bank_old = "MemBank_B"
        layer_params_old = None

        for layer_params in layer_parameters:
            print(layer_params)
            if params["langs"] == "c" and params["precision"] == "fix16":
                if layer_params["layer_name"].find("input") != -1:
                    f.write(str("int network(int16_t* input_data, int16_t* output_data){{\n"))
                    f.write("\tfor(int i = 0; i < {0}_depth * {0}_height * {0}_width; i++){{\n".format(layer_params["layer_name"]))
                    f.write("\t\t{0}[i] = input_data[i];\n".format(Memory_Bank))
                    f.write("\t}\n")

                elif layer_params["layer_name"].find("Padding2D") != -1:
                    f.write(str("\tpadding2d_fix16({}, {},\n\t").format(layer_params["padding_h"], layer_params["padding_w"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_height, {0}_width, (int16_t*) {1});\n\n").format(layer_params["layer_name"], Memory_Bank))

                elif layer_params["layer_name"].find("MaxPooling2D") != -1:
                    f.write(str("\tmax_pooling2d_fix16({},\n\t").format(layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1});\n\n").format(layer_params["layer_name"], Memory_Bank))

                elif layer_params["layer_name"].find("UpSampling2D") != -1:
                    f.write(str("\tup_sampling2d_fix16({},\n\t").format(layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1});\n\n").format(layer_params["layer_name"], Memory_Bank))

                elif layer_params["layer_name"].find("DepthwiseConv2D") != -1:
                    if layer_params["activation"] == "relu":
                        relu_flag = 1
                    else:
                        relu_flag = 0
                    f.write(str("\tsdepthwise_conv2d_fix16({0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params["layer_name"], Memory_Bank))
                    f.write(str("(int16_t*) {0}_b_d,\n\t").format(layer_params["layer_name"]))
                    f.write(str("{0}, {1}, (int16_t*) {2}_w_d, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))

                elif layer_params["layer_name"].find("SeparableConv2D") != -1:
                    if layer_params["activation"] == "relu":
                        relu_flag = 1
                    else:
                        relu_flag = 0

                    if Memory_Bank == "MemBank_A":
                        Memory_Bank_Out = "MemBank_B"
                    else:
                        Memory_Bank_Out = "MemBank_A"

                    f.write(str("\tdepthwise_conv2d_fix16({0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params["layer_name"], Memory_Bank))
                    f.write(str("(int16_t*) {0}_b_d,\n\t").format(layer_params["layer_name"]))
                    f.write(str("{0}, {1}, (int16_t*) {2}_w_d, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], 0))

                    f.write(str("\tpointwise_conv2d_fix16({1}_depth, {0}_height, {0}_width, (int16_t*) {2},\n\t").format(layer_params["layer_name"], layer_params_old["layer_name"], Memory_Bank))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params["layer_name"], Memory_Bank_Out))
                    f.write(str("(int16_t*) {0}_b_p,\n\t").format(layer_params["layer_name"]))
                    f.write(str("1, 1, (int16_t*) {0}_w_p, {1}, fractal_width_{0});\n\n").format(layer_params["layer_name"], relu_flag))
                    Memory_Bank = Memory_Bank_Out

                elif layer_params["layer_name"].find("Conv2D") != -1:
                    if layer_params["activation"] == "relu":
                        relu_flag = 1
                    else:
                        relu_flag = 0
                    f.write(str("\tconv2d_fix16({0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, (int16_t*) {1},\n\t").format(layer_params["layer_name"], Memory_Bank))
                    f.write(str("(int16_t*) {}_b,\n\t").format(layer_params["layer_name"]))
                    f.write(str("{0}, {1}, (int16_t*) {2}_w, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))

                layer_params_old = layer_params.copy()
                Memory_Bank_old = Memory_Bank
                if Memory_Bank == "MemBank_A":
                    Memory_Bank = "MemBank_B"
                else:
                    Memory_Bank = "MemBank_A"

            f.write("void main(void){{")
            f.write('\tFILE* fp = fopen("template_input_fix16.tsv", "w");\n\t')
            f.write("array_fprintf_2D_fix16(input_0_height, input_0_width, input_0_array[0], '\\t', fp, fractal_width_input_0);\n\t")
            f.write("fclose(fp);\n\n")
            f.write("}\n")


if __name__ == "__main__":

    for folder in folders:
        for lang in languages:
            if os.path.isdir(str("./{}_{}").format(folder, lang)) is False:
                os.mkdir(str("./{}_{}").format(folder, lang))

    layer_layer_params, itr_counter = parse_keras_model("keras_mnist_DCAE/keras_mnist_DCAE.json", "keras_mnist_DCAE/keras_mnist_DCAE.h5")

    for langs in languages:
        for folder in folders:
            for precise in precision:
                # "./arrays_C/arrays_float32.h"
                buff_name = str("./{0}_{1}/{0}_{2}.{1}").format(folder, langs, precise)
                print(buff_name[:-2])
                template_writer(layer_layer_params, buff_name[:-2])

                # print(buff_name)

    # for layer_p in layer_layer_params:
    #     print(layer_p["layer_name"], layer_p["depth"], layer_p["height"], layer_p["width"], end=" ")
    #     if(layer_p["layer_name"].find("Conv2D") != -1):
    #         print(layer_p["ksize_h"], layer_p["ksize_w"], layer_p["bias_length"])
    #     else:
    #         print()

    #     if(layer_p["layer_name"] == "input_0"):
    #         max_array_size = layer_p["depth"] * layer_p["height"] * layer_p["width"]
    #     if(layer_p["depth"] * layer_p["height"] * layer_p["width"] > max_array_size):
    #         max_array_size = layer_p["depth"] * layer_p["height"] * layer_p["width"]

    # print(max_array_size)
    # print(itr_counter)


"""
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

        for i in layer_layer_params:
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

        for i in layer_layer_params:
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
"""
