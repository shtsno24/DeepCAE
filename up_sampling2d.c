#include <stdint.h>
#include "up_sampling2d.h"


uint8_t up_sampling2d(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_depth],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_depth]){

    // output_* "must" be Divisible by kernel_size on any axis
    
    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                for(uint16_t kernel_h = 0; kernel_h < kernel_size; kernel_h++){
                    for(uint16_t kernel_w = 0; kernel_w < kernel_size; kernel_w++){
                            output[out_d][out_h][out_w] = input[out_d][out_h/kernel_size + kernel_h][out_w/kernel_size + kernel_w];
                    }
                }
            }
        }
    }
    return(0);
}