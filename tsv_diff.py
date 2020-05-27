import numpy as np
from keras_mnist_DCAE import float2fixed
import os


# source_file = input("source_file name : ")
# source_file = "CPU/C/template_output_float32_c_Sep"
source_file = "keras_mnist_DCAE/keras_mnist_DCAE_output"
source_array = np.loadtxt(source_file + ".tsv", dtype=np.float32, delimiter='\t')
source_array = source_array.reshape((1,) + source_array.shape)
source_array = (source_array - np.min(source_array)) / (np.max(source_array) - np.min(source_array))
# source_array = (source_array * 255).astype(np.uint8)
source_array = (source_array * 255).astype(np.float32)
print(source_array.shape, source_array.dtype, np.max(source_array), np.min(source_array),"\n-------------------")


# target_file = input("target_file name : ")
# target_file = "CPU/C/template_output_fix16_c_Sep"
target_file = "CPU/C/template_output_float32_c_Sep"
target_array = np.loadtxt(target_file + ".tsv", dtype=np.float32, delimiter='\t')
target_array = target_array.reshape((1,) + target_array.shape)
target_array = (target_array - np.min(target_array)) / (np.max(target_array) - np.min(target_array))
# target_array = (target_array * 255).astype(np.uint8)
target_array = (target_array * 255).astype(np.float32)
print(target_array.shape, target_array.dtype, np.max(target_array), np.min(target_array), "\n-------------------")


diff_array = np.sqrt((source_array - target_array)**2).astype(np.float32)
print(diff_array.shape, diff_array.dtype, "\n-------------------")
print("max diff : ", np.max(diff_array))
print("min diff : ", np.min(diff_array))
print("mean diff : ", np.mean(diff_array))


np.savetxt("diff_Python_fp32_float32.tsv".format(source_file, target_file), diff_array[0], delimiter="\t", fmt='%e')
input(">>>")