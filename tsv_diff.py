import numpy as np
from keras_mnist_DCAE import float2fixed
import os


# source_file = input("source_file name : ")
source_file = "CPU/template_output_float32_c_Sep"
source_array = np.loadtxt(source_file + ".tsv", dtype=np.float32, delimiter='\t')
source_array = source_array.reshape((1,) + source_array.shape)
print(source_array.shape, source_array.dtype, "\n-------------------")

# target_file = input("target_file name : ")
target_file = "CPU/template_output_fix16_c_Sep"
target_array = np.loadtxt(target_file + ".tsv", dtype=np.float32, delimiter='\t')
target_array = target_array.reshape((1,) + target_array.shape)
print(target_array.shape, target_array.dtype, "\n-------------------")


diff_array = np.sqrt((source_array - target_array)**2)
print(diff_array.shape, diff_array.dtype, "\n-------------------")
print("max diff : ", np.max(diff_array))
print("min diff : ", np.min(diff_array))
print("mean diff : ", np.mean(diff_array))


np.savetxt("diff_fp32_fix16.tsv".format(source_file, target_file), diff_array[0], delimiter="\t", fmt='%e')