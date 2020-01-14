import cv2
import numpy as np
from sklearn import datasets

mnist = datasets.fetch_openml('mnist_784', version=1,)
mnist_data = mnist.data

print(mnist_data.shape, mnist_data.dtype)

mnist_img = mnist_data[0].astype(np.uint8)
mnist_img = mnist_img.reshape((28, 28))

ret = cv2.imwrite("./mnist_data_img.png", mnist_img)

if ret:
    print("Done")
else:
    print("meow")
