import numpy as np
from keras_mnist_DCAE import float2fixed
import keras_weight_generator
import os

if os.path.isdir("test_data") is False:
    os.mkdir("test_data")

tsv_file = input("file name : ")
tsv_array = np.loadtxt(tsv_file + ".tsv", dtype=np.float32, delimiter='\t')
tsv_array = tsv_array.reshape((1,) + tsv_array.shape)
print(tsv_array.shape, tsv_array.dtype)

tsv_array_fix16 = float2fixed.float2fixed_array(np.int16, 16-2, tsv_array)
print(tsv_array_fix16.shape, tsv_array_fix16.dtype, tsv_array_fix16)

with open("test_data/test_data.h", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(str("int16_t test_input_fix16[{}][{}][{}] = ").format(tsv_array_fix16.shape[0], tsv_array_fix16.shape[1], tsv_array_fix16.shape[2]))
    keras_weight_generator.write_array_ND(tsv_array_fix16, f)
    f.write(";\n\n")

    f.write(str("float test_input_float32[{}][{}][{}] = ").format(tsv_array.shape[0], tsv_array.shape[1], tsv_array.shape[2]))
    keras_weight_generator.write_array_ND(tsv_array, f)
    f.write(";\n\n")
