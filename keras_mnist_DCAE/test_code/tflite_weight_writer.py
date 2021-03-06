from __future__ import print_function
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import OrderedDict
import json
import datetime


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


def write_weight_Conv2D(weight, bias, file_name, weight_array_name, bias_array_name):
    # Weight generated by Keras has 4D array with (height, width, input_depth, output_depth)
    # Transpose weight axis from (height, width, input_depth, output_depth) to (output_depth, input_depth ,height, width)

    with open(file_name, 'w') as f:

        # reshape weight array
        weight = weight.transpose(3,2,0,1)

        # headers
        todaytime = str(datetime.datetime.today())
        f.write("/*\r\n")
        f.write(" * author : shtsno24\r\n")
        f.write(" * Date : " + todaytime + "\r\n")
        f.write(" *\n */\n")

        # weights
        f.write("const float " + weight_array_name)
        f.write(str("[{}][{}][{}][{}] =\r\n").format(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]))
        write_array_ND(weight, f)
        f.write(";")
        f.write("\r\n\r\n")

        # bias
        f.write("const float " + bias_array_name)
        f.write("[%d] = " % bias.shape)
        write_array_ND(bias, f)
        f.write(";")
        f.write("\r\n")


if __name__ == "__main__":

    interpreter = tf.lite.Interpreter(model_path="tflite_mnist_DCAE.tflite")
    interpreter.allocate_tensors()
    all_tensors = interpreter.get_tensor_details()
    for tensor in all_tensors:
        print(tensor['name'], 
            type(interpreter.get_tensor(tensor['index'])), 
            interpreter.get_tensor(tensor['index']).dtype,
            interpreter.get_tensor(tensor['index']).shape)
        # print(interpreter.get_tensor(tensor['index']))

    # with open("keras_mnist_DCAE.json") as jfile:
    #     model = load_model("keras_mnist_DCAE.h5")
    #     model.summary()
    #     model_weights_itr = model.get_weights()
    #     model_layers_itr = json.load(jfile, object_pairs_hook=OrderedDict)

    #     itr_counter = 0
    #     for layers in model_layers_itr["config"]["layers"]:
    #         print()
    #         print(layers["class_name"])
    #         if(layers["class_name"].find("UpSampling") == -1 and layers["class_name"].find("MaxPooling2D") == -1):
    #             param_w = model_weights_itr[itr_counter]
    #             param_b = model_weights_itr[itr_counter + 1]
    #             itr_counter += 2
    #             print("weight", param_w.shape, len(param_w.shape))
    #             print("bias", param_b.shape, len(param_b.shape))
    #             write_weight_Conv2D(param_w, 
    #                                 param_b, 
    #                                 layers["class_name"] + "_" + str(int(itr_counter/2) - 1) + ".h", 
    #                                 layers["class_name"] + "_" + str(int(itr_counter/2) - 1) + "_w", 
    #                                 layers["class_name"] + "_" + str(int(itr_counter/2) - 1) + "_b")
    #         else:
    #             print("This Layer has no Parameter")
