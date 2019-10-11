import numpy as np

def float2fixed_norm(max_val, min_val, fractal_width, fval):
    """
        max_val, min_val : float
    """
    if(fval > max_val):
        fval = max_val
    elif(fval < min_val):
        fval = min_val

    inorm = (fval - min_val)/(max_val - min_val) * (2 ** fractal_width)
    return int(inorm)


def fixed2float_norm(max_val, min_val, fractal_width, inorm): 
    
    """
        max_val, min_val : float
    """
    fval = (float(inorm) / (2.0 ** fractal_width)) * float(max_val - min_val)
    return fval


def float2fixed_norm_array(max_val, min_val, array_type, fractal_width, farray):
    """
        max_val, min_val    : float
        array_type          : target type
    """
    farray[farray > max_val] = max_val
    farray[farray < min_val] = min_val
    
    iarray = (farray - min_val)/(max_val - min_val) * (2 ** fractal_width)
    return iarray


def fixed2floar_norm_array(max_val, min_val, array_type, fractal_width, iarray):
    """
        max_val, min_val    : float
        array_type          : target type
    """
    farray = (iarray.astype(array_type) / (2.0 ** fractal_width)) * float(max_val - min_val)
    return farray


def float2fixed(fractal_width, fval):
    return int(fval * (2 ** fractal_width))


def fixed2float(fractal_width, ival):
    return float(ival) / (2.0 ** fractal_width)


def float2fixed_array(array_type, fractal_width, farray):
    iarray = (farray * (2 ** fractal_width)).astype(array_type)
    return iarray


def fixed2floar_array(array_type, fractal_width, iarray):
    farray = (iarray.astype(array_type) / (2.0 ** fractal_width))
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
