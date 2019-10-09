#include <stdint.h>
#include "conv2d.h"


uint8_t conv2d(uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t input[input_depth][input_height][input_width],
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t output[output_depth][output_height][output_width],
int16_t bias[output_depth],
uint16_t kernel_height, uint16_t kernel_width, int16_t kernel[output_depth][input_depth][kernel_height][kernel_width],
uint8_t relu){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1


    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                output[out_d][out_h][out_w] = 0;
                for(uint16_t in_d = 0; in_d < input_depth; in_d++){
                    for(uint16_t k_h = 0; k_h < kernel_height; k_h++){
                        for(uint16_t k_w = 0; k_w < kernel_width; k_w++){
                            output[out_d][out_h][out_w] += 
                                input[in_d][out_h + k_h][out_w + k_w] * 
                                kernel[out_d][in_d][k_h][k_w];
                        }
                    }
                }
                output[out_d][out_h][out_w] += bias[out_d];

                if(relu == 1){
                    if(output[out_d][out_h][out_w] < 0){
                        output[out_d][out_h][out_w] = 0;
                    }
                }
            }
        }
    }
    return(0);
}
