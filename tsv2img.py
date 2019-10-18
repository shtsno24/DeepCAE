import numpy as np
from PIL import Image

tsv_file = input("file name : ")
tsv_array = np.loadtxt(tsv_file, dtype=np.float64, delimiter='\t')
print(tsv_array.shape, tsv_array.dtype)

tsv_array = (tsv_array - np.min(tsv_array)) / (np.max(tsv_array) - np.min(tsv_array)) * 255

pil_img = Image.fromarray(tsv_array.astype(np.uint8))
pil_img.save(tsv_file + ".png")
