from __future__ import print_function
from tensorflow.keras.models import load_model
from collections import OrderedDict
import json
import numpy as np


def parse_keras_model(json_file, h5_file):
    # with open("keras_mnist_DCAE/keras_mnist_DCAE.json") as jfile:
    with open(json_file) as jfile:

        # model = load_model("keras_mnist_DCAE/keras_mnist_DCAE.h5")
        model = load_model(h5_file)
        model.summary()
        model_arrays_itr = json.load(jfile, object_pairs_hook=OrderedDict)

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
    return layer_params, itr_counter
