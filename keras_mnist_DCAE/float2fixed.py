import numpy as np


def float2fixed(fractal_width, f):
    return int(f * (2**fractal_width))


def fixed2float(fractal_width, i):
    return float(i / (2**fractal_width))


def float2fixed_array(array_type, fractal_width, farray):
    iarray = (farray * (2 ** fractal_width)).astype(array_type)
    return iarray


def fixed2floar_array(array_type, fractal_width, iarray):
    farray = (iarray.astype(array_type) / (2 ** fractal_width))
    return farray


if __name__ == "__main__":
    float_array = np.array([[x for x in range(10)] for y in range(3)], dtype=np.float32)
    print(float_array.dtype, float_array.shape)
    float_array *= 1.0 / 5.0
    float_array[1:2, :] *= -1
    print(float_array)

    int_array = float2fixed_array(np.int16, 8, float_array)
    print(int_array.dtype, int_array.shape)
    print(np.vectorize(np.binary_repr)(int_array, width=8))

    del(float_array)
    float_array = fixed2floar_array(np.float32, 8, int_array)
    print(float_array.dtype, float_array.shape)
    print(float_array)
