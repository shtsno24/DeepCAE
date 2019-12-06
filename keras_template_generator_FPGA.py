from __future__ import print_function
import datetime
import os

from keras_model_parser import parse_keras_model

languages = ["c", "cpp"]
folders = ["template"]
precision = ["fix16", "float32"]


def template_writer(layer_parameters, file_name):

    params = {"langs": "cpp", "precision": "fix16"}
    itr_counter = {"MaxPooling2D": 0, "UpSampling2D": 0, "Conv2D": 0, "Padding2D": 0, "SeparableConv2D": 0, "DepthwiseConv2D": 0, "PointwiseConv2D": 0, "input_0": 1}

    if file_name.find("cpp") != -1:
        params["langs"] = "cpp"
    else:
        params["langs"] = "c"

    if file_name.find("float") != -1:
        params["precision"] = "float32"
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
        
        if params["langs"] == "c":
            f.write("#include <stdint.h>\n")
            f.write("#include <stdio.h>\n\n")
        else:
            f.write("#include <cstdint>\n")
            f.write("#include <vector>\n")
            f.write("#include <iostream>\n")
            f.write("#include <fstream>\n\n")
            f.write("using namespace std;\n\n")

        f.write('#include "./../test_data/test_data.h"\n')
        f.write('#include "./../layers_{0}/array_printf_{1}.h"\n'.format(params["langs"], params["precision"]))
        f.write('#include "./../arrays_{0}/arrays_{1}.h"\n'.format(params["langs"], params["precision"]))
        f.write('#include "./../layers_{0}/layers.h"\n'.format(params["langs"]))
        f.write('#include "./../weights_{0}/weights_{1}.h"\n\n'.format(params["langs"], params["precision"]))

        Memory_Bank = "MemBank_A"
        Memory_Bank_old = "MemBank_B"
        layer_params_old = None
        array_type = None

        for layer_params in layer_parameters:
            print(layer_params)
            if params["precision"] == "fix16":
                array_type = "int16_t"
            else:
                array_type = "float"

            if layer_params["layer_name"].find("input") != -1:
                f.write(str("int network({0}* input_data, {0}* output_data){{\n").format(array_type))

                if params["langs"] == "c":
                    f.write("\n\t{} MemBank_A[max_array_size], MemBank_B[max_array_size];\n".format(array_type))
                    f.write("\tfor(int i = 0; i < {0}_depth * {0}_height * {0}_width; i++){{\n".format(layer_params["layer_name"]))
                    f.write("\t\t{0}[i] = input_data[i];\n".format(Memory_Bank))
                    f.write("\t}\n")
                else:
                    f.write("\tint i = 0;\n")
                    f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(layer_params["layer_name"]))
                    f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(layer_params["layer_name"]))
                    f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(layer_params["layer_name"]))
                    f.write("\t\t\t\t{0}[depth][height][width] = input_data[i];\n".format(Memory_Bank))
                    f.write("\t\t\t\ti += 1;\n")
                    f.write("\t\t\t}\n")
                    f.write("\t\t}\n")
                    f.write("\t}\n\n")

            elif layer_params["layer_name"].find("Padding2D") != -1:
                if params["langs"] == "c":
                    f.write(str("\tpadding2d_{0}_{1}({2}, {3},\n\t").format(params["precision"], itr_counter["Padding2D"], layer_params["padding_h"], layer_params["padding_w"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_height, {0}_width, ({1}*){2});\n\n").format(layer_params["layer_name"], array_type, Memory_Bank))
                else:
                    f.write(str("\tpadding2d_{0}({1}, {2},\n\t").format(params["precision"], layer_params["padding_h"], layer_params["padding_w"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_height, {0}_width, {1});\n\n").format(layer_params["layer_name"], Memory_Bank))

                itr_counter["Padding2D"] += 1

            elif layer_params["layer_name"].find("MaxPooling2D") != -1:
                if params["langs"] == "c":
                    f.write(str("\tmax_pooling2d_{0}_{1}({2},\n\t").format(params["precision"], itr_counter["MaxPooling2D"], layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2});\n\n").format(layer_params["layer_name"], array_type, Memory_Bank))
                else:
                    f.write(str("\tmax_pooling2d_{0}({1},\n\t").format(params["precision"], layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1},\n\t").format(layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1});\n\n").format(layer_params["layer_name"], Memory_Bank))

                itr_counter["MaxPooling2D"] += 1

            elif layer_params["layer_name"].find("UpSampling2D") != -1:
                if params["langs"] == "c":
                    f.write(str("\tup_sampling2d_{0}_{1}({2},\n\t").format(params["precision"], itr_counter["UpSampling2D"], layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2});\n\n").format(layer_params["layer_name"], array_type, Memory_Bank))
                else:
                    f.write(str("\tup_sampling2d_{0}({1},\n\t").format(params["precision"], layer_params["ksize_h"]))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {2},\n\t").format(layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {2});\n\n").format(layer_params["layer_name"], array_type, Memory_Bank))

            elif layer_params["layer_name"].find("DepthwiseConv2D") != -1:
                if layer_params["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0
                if params["langs"] == "c":
                    f.write(str("\tdepthwise_conv2d_{0}({1}_depth, {1}_height, {1}_width, ({2}*){3},\n\t").format(params["precision"], layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params["layer_name"], array_type, Memory_Bank))
                    f.write(str("({1}*) {0}_b,\n\t").format(layer_params["layer_name"], array_type))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, ({4}*) {2}_w, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag, array_type))
                    else:
                        f.write(str("{0}, {1}, ({4}*) {2}_w, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag, array_type))
                else:
                    f.write(str("\tdepthwise_conv2d_{0}({1}_depth, {1}_height, {1}_width, {2},\n\t").format(params["precision"], layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1},\n\t").format(layer_params["layer_name"], Memory_Bank))
                    f.write(str("{0}_b,\n\t").format(layer_params["layer_name"]))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, {2}_w, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))
                    else:
                        f.write(str("{0}, {1}, {2}_w, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))

            elif layer_params["layer_name"].find("SeparableConv2D") != -1:
                if layer_params["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0

                if Memory_Bank == "MemBank_A":
                    Memory_Bank_Out = "MemBank_B"
                else:
                    Memory_Bank_Out = "MemBank_A"

                if params["langs"] == "c":
                    f.write(str("\tdepthwise_conv2d_{0}({1}_depth, {1}_height, {1}_width, ({2}*){3},\n\t").format(params["precision"], layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {1}_height, {1}_width, ({2}*){3},\n\t").format(layer_params_old["layer_name"], layer_params["layer_name"], array_type, Memory_Bank))
                    f.write(str("({1}*) {0}_b_d,\n\t").format(layer_params["layer_name"], array_type))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, ({4}*) {2}_w_d, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], 0, array_type))
                    else:
                        f.write(str("{0}, {1}, ({4}*) {2}_w_d, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], 0, array_type))
                else:
                    f.write(str("\tdepthwise_conv2d_{0}({1}_depth, {1}_height, {1}_width, {2},\n\t").format(params["precision"], layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {1}_height, {1}_width, {2},\n\t").format(layer_params_old["layer_name"], layer_params["layer_name"], Memory_Bank))
                    f.write(str("{0}_b_d,\n\t").format(layer_params["layer_name"]))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, {2}_w_d, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], 0))
                    else:
                        f.write(str("{0}, {1}, {2}_w_d, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], 0))

                if params["langs"] == "c":
                    f.write(str("\tpointwise_conv2d_{0}({2}_depth, {1}_height, {1}_width, ({3}*){4},\n\t").format(params["precision"], layer_params["layer_name"], layer_params_old["layer_name"], array_type, Memory_Bank))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params["layer_name"], array_type, Memory_Bank_Out))
                    f.write(str("({1}*){0}_b_p,\n\t").format(layer_params["layer_name"], array_type))
                    if params["precision"] == "fix16":
                        f.write(str("1, 1, ({2}*){0}_w_p, {1}, fractal_width_{0});\n\n").format(layer_params["layer_name"], relu_flag, array_type))
                    else:
                        f.write(str("1, 1, ({2}*){0}_w_p, {1});\n\n").format(layer_params["layer_name"], relu_flag, array_type))
                else:
                    f.write(str("\tpointwise_conv2d_{0}({2}_depth, {1}_height, {1}_width, {3},\n\t").format(params["precision"], layer_params["layer_name"], layer_params_old["layer_name"], Memory_Bank))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1},\n\t").format(layer_params["layer_name"], Memory_Bank_Out))
                    f.write(str("{0}_b_p,\n\t").format(layer_params["layer_name"]))
                    if params["precision"] == "fix16":
                        f.write(str("1, 1, {0}_w_p, {1}, fractal_width_{0});\n\n").format(layer_params["layer_name"], relu_flag))
                    else:
                        f.write(str("1, 1, {0}_w_p, {1});\n\n").format(layer_params["layer_name"], relu_flag))

                Memory_Bank = Memory_Bank_Out

            elif layer_params["layer_name"].find("Conv2D") != -1:
                if layer_params["activation"] == "relu":
                    relu_flag = 1
                else:
                    relu_flag = 0

                if params["langs"] == "c":
                    f.write(str("\tconv2d_{0}({1}_depth, {1}_height, {1}_width, ({2}*){3},\n\t").format(params["precision"], layer_params_old["layer_name"], array_type, Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, ({1}*){2},\n\t").format(layer_params["layer_name"], array_type, Memory_Bank))
                    f.write(str("({1}*) {0}_b,\n\t").format(layer_params["layer_name"], array_type))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, ({4}*) {2}_w, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag, array_type))
                    else:
                        f.write(str("{0}, {1}, ({4}*) {2}_w, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag, array_type))
                else:
                    f.write(str("\tconv2d_{0}({1}_depth, {1}_height, {1}_width, {2},\n\t").format(params["precision"], layer_params_old["layer_name"], Memory_Bank_old))
                    f.write(str("{0}_depth, {0}_height, {0}_width, {1},\n\t").format(layer_params["layer_name"], Memory_Bank))
                    f.write(str("{0}_b,\n\t").format(layer_params["layer_name"]))
                    if params["precision"] == "fix16":
                        f.write(str("{0}, {1}, {2}_w, {3}, fractal_width_{2});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))
                    else:
                        f.write(str("{0}, {1}, {2}_w, {3});\n\n").format(layer_params["ksize_h"], layer_params["ksize_w"], layer_params["layer_name"], relu_flag))

            layer_params_old = layer_params.copy()
            Memory_Bank_old = Memory_Bank
            if Memory_Bank == "MemBank_A":
                Memory_Bank = "MemBank_B"
            else:
                Memory_Bank = "MemBank_A"

        # if params["langs"] == "c" and params["precision"] == "fix16":
        if params["langs"] == "c":
            f.write("\tfor(int i = 0; i < {0}_depth * {0}_height * {0}_width; i++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\toutput_data[i] = {0}[i];\n".format(Memory_Bank_old))
            f.write("\t}\n\n")
            f.write("\treturn(0);\n\n")
            f.write("}\n\n")

            f.write("int main(void){\n")
            f.write("\t{0} output_buffer[{1}][{2}][{3}];\n\n".format(array_type, "1", "28", "28"))

            f.write("\tnetwork(({0}*)test_input_{1}, ({0}*)output_buffer);\n\n".format(array_type, params["precision"]))
            f.write('\tFILE* fp = fopen("template_input_{0}.tsv", "w");\n\t'.format(params["precision"]))
            if params["precision"] == "fix16":
                f.write("array_fprintf_2D_{0}(input_0_height, input_0_width, test_input_{0}[0], '\\t', fp, fractal_width_input_0);\n\t".format(params["precision"]))
                f.write("fclose(fp);\n\n")

                f.write('\tfp = fopen("template_output_{0}.tsv", "w");\n\t'.format(params["precision"]))
                f.write("array_fprintf_2D_{0}({1}_height, {1}_width, output_buffer[0], '\\t', fp, fractal_width_{1});\n\t".format(params["precision"], layer_params_old["layer_name"]))
            else:
                f.write("array_fprintf_2D_{0}(input_0_height, input_0_width, test_input_{0}[0], '\\t', fp);\n\t".format(params["precision"]))
                f.write("fclose(fp);\n\n")

                f.write('\tfp = fopen("template_output_{0}.tsv", "w");\n\t'.format(params["precision"]))
                f.write("array_fprintf_2D_{0}({1}_height, {1}_width, output_buffer[0], '\\t', fp);\n\t".format(params["precision"], layer_params_old["layer_name"]))
            f.write("fclose(fp);\n\t")
            f.write("return(0);\n\n")
            f.write("}\n")
        else:
            f.write("\ti = 0;\n")
            f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\t\t\toutput_data[i] = {0}[depth][height][width];\n".format(Memory_Bank_old))
            f.write("\t\t\t\ti += 1;\n")
            f.write("\t\t\t}\n")
            f.write("\t\t}\n")
            f.write("\t}\n\n")
            f.write("\treturn(0);\n\n")
            f.write("}\n\n")

            f.write("int main(void){\n")
            f.write("\t{0} output_buffer[{1}][{2}][{3}];\n".format(array_type, "1", "28", "28"))
            f.write("\tvector< vector< vector< {0}> > > input_img({1}, vector< vector< {0}> >({2}, vector< {0}>({3})));\n".format(array_type, 1, 28, 28))
            f.write("\tvector< vector< vector< {0}> > > output_img({1}, vector< vector< {0}> >({2}, vector< {0}>({3})));\n\n".format(array_type, 1, 28, 28))

            # f.write("\tint i = 0;")
            f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format("input_0"))
            f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format("input_0"))
            f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format("input_0"))
            f.write("\t\t\t\tinput_img[depth][height][width] = test_input_{0}[depth][height][width];\n".format(params["precision"]))
            # f.write("\t\t\t\t{0}[depth][height][width] = output_buffer[depth][height][width];\n".format("input_img"))
            # f.write("\t\t\t\ti += 1;\n")
            f.write("\t\t\t}\n")
            f.write("\t\t}\n")
            f.write("\t}\n\n")

            f.write("\tnetwork(({0}*)test_input_{1}, ({0}*)output_buffer);\n\n".format(array_type, params["precision"]))

            # f.write("\tint i = 0;")
            f.write("\tfor(int depth = 0; depth < {0}_depth; depth++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\tfor(int height = 0; height < {0}_height; height++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\t\tfor(int width = 0; width < {0}_width; width++){{\n".format(layer_params_old["layer_name"]))
            f.write("\t\t\t\t{0}[depth][height][width] = output_buffer[depth][height][width];\n".format("output_img"))
            # f.write("\t\t\t\ti += 1;\n")
            f.write("\t\t\t}\n")
            f.write("\t\t}\n")
            f.write("\t}\n")

            # ofstream fp("template_input_fix16_Sep.tsv");
            # array_fprintf_2D_fix16(28, 28, input_img[0], '\t', fp, fractal_width_input_0);
            # fp.close();

            f.write('\tofstream fp("template_input_{0}.tsv");\n\t'.format(params["precision"]))
            if params["precision"] == "fix16":
                f.write("array_fprintf_2D_{0}(input_0_height, input_0_width, input_img[0], '\\t', fp, fractal_width_input_0);\n\t".format(params["precision"]))
                f.write("fp.close();\n\n")

                f.write('\tfp.open("template_output_{0}.tsv");\n\t'.format(params["precision"]))
                f.write("array_fprintf_2D_{0}({1}_height, {1}_width, output_img[0], '\\t', fp, fractal_width_{1});\n\t".format(params["precision"], layer_params_old["layer_name"]))
            else:
                f.write("array_fprintf_2D_{0}(input_0_height, input_0_width, input_img[0], '\\t', fp);\n\t".format(params["precision"]))
                f.write("fp.close();\n\n")

                f.write('\tfp.open("template_output_{0}.tsv");\n\t'.format(params["precision"]))
                f.write("array_fprintf_2D_{0}({1}_height, {1}_width, output_img[0], '\\t', fp);\n\t".format(params["precision"], layer_params_old["layer_name"]))
            f.write("fp.close();\n\t")
            f.write("return(0);\n\n")
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
                print(buff_name)
                template_writer(layer_layer_params, buff_name)
