#include <stdint.h>
#include "conv2d.h"

uint8_t conv2d(uint16_t filters, ARRAY_PARAMS_3D kernel_shape, int16_t* kernel,
ARRAY_PARAMS_3D input_shape, int16_t* input, uint16_t pad, uint16_t stride,
ARRAY_PARAMS_3D bias_shape, int16_t* bias,
ARRAY_PARAMS_3D output_shape, int16_t* output){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1

    for(uint16_t out_d = 0; out_d < output_shape.depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_shape.height; out_h++){
            for(uint16_t out_w = 0; out_w < output_shape.width; out_w++){
                output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] = 0;
                for(uint16_t in_d = 0; in_d < input_shape.depth; in_d++){
                    for(uint16_t k_h = 0; k_h < kernel_shape.height; k_h++){
                        for(uint16_t k_w = 0; k_w < kernel_shape.width; k_w++){
                            output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] += 
                                input[in_d * input_shape.height * input_shape.width + (out_h + k_h) * output_shape.width + (out_w + k_w)] * 
                                kernel[in_d * kernel_shape.height * kernel_shape.width + k_h * kernel_shape.width + k_w];
                        }
                    }
                }
                output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] += bias[out_d];
            }
        }
    }
    return(0);
}
