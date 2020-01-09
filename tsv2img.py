import numpy as np
from PIL import Image

tsv_file = input("file name : ")
tsv_array = np.loadtxt(tsv_file + ".tsv", dtype=np.float32, delimiter='\t')
print(tsv_array.shape, tsv_array.dtype)
print("max : {}\nmin : {}".format(np.max(tsv_array), np.min(tsv_array)))
tsv_array = (tsv_array - np.min(tsv_array)) / (np.max(tsv_array) - np.min(tsv_array)) * 255

pil_img = Image.fromarray(tsv_array.astype(np.uint8))
pil_img.save(tsv_file + ".png")
input(">>>")