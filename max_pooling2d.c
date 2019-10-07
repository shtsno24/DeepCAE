#include <stdint.h>
#include "conv2d.h"

uint8_t max_pooling2d(uint16_t kernel_size,
ARRAY_PARAMS_3D input_shape, int16_t* input, uint16_t pad, uint16_t stride,
ARRAY_PARAMS_3D output_shape, int16_t* output){
    
    // input shape *must* be Divisible by kernel_size on any axis
    for(uint16_t out_d = 0; out_d < output_shape.depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_shape.height; out_h){
            for(uint16_t out_w = 0; out_w < output_shape.width; out_w){
                output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] = 0;
                for(uint16_t in_h = 0; in_h < kernel_size; in_h++){
                    for(uint16_t in_w = 0; in_w < kernel_size; in_w++){
                        if (output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] < 
                        input[out_d * output_shape.height * output_shape.width + (out_h*kernel_size + in_h) * output_shape.width + (out_w*kernel_size + in_w)]){
                            output[out_d * output_shape.height * output_shape.width + out_h * output_shape.width + out_w] = 
                            input[out_d * output_shape.height * output_shape.width + (out_h*kernel_size + in_h) * output_shape.width + (out_w*kernel_size + in_w)];
                        }
                    }
                }
            }
        }
    }
    return(0);
}
