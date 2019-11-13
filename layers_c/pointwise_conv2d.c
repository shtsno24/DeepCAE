#include <stdint.h>
#include "pointwise_conv2d.h"
#include <stdio.h>


uint8_t pointwise_conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, int16_t* input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, int16_t* output,
const int16_t* bias,
uint16_t kernel_height, uint16_t kernel_width, const int16_t* kernel,
uint8_t relu, uint8_t fractal_width, uint8_t debug){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1
    long int addr_diff = 0;
    FILE *fp;
    if(debug == 1){
        fp = fopen("debug.txt", "a");
        fprintf(fp, "============\r\n");
        fprintf(fp, "input @ 0x%X\r\n", input);
    }

    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                output[out_d * output_height * output_width + out_h * output_width + out_w] = 0;
                for(uint16_t in_d = 0; in_d < input_depth; in_d++){
                    //output[out_d][out_h][out_w] += input[in_d][out_h][out_w] * kernel[out_d][in_d][1][1];
                    output[out_d * output_height * output_width + out_h * output_width + out_w] += 
                        (int16_t)(((int32_t)(input[in_d * input_height * input_width + out_h * input_width + out_w]) * 
                                    (int32_t)(kernel[out_d * input_depth + in_d])) >> fractal_width);
                    if(debug == 1){
                        addr_diff = &(input[in_d * input_height * input_width + out_h * input_width + out_w ]) - &(input[0]);
                        fprintf(fp, "pointwise2d : input : % 5d\r\n", input[in_d * input_height * input_width + out_h * input_width + out_w]);
                        fprintf(fp, "pointwise2d : input : 0x%X\r\n", addr_diff);
                        // fprintf(fp, "pointwise2d : kernel : % 5d\r\n", kernel[out_d * input_depth + in_d]);
                        // fprintf(fp, "pointwise2d : output += % 5d\r\n=======\r\n", input[in_d * input_height * input_width + out_h * input_width + out_w] * kernel[out_d * input_depth + in_d] >> fractal_width);
                    }
                }

                output[out_d * output_height * output_width + out_h * output_width + out_w] += bias[out_d];

                if(relu == 1){
                    if(output[out_d * output_height * output_width + out_h * output_width + out_w] < 0){
                        output[out_d * output_height * output_width + out_h * output_width + out_w] = 0;
                    }
                }
            }
        }
    }
    if(debug == 1){
        fclose(fp);
    }
    return(0);
}

uint8_t pointwise_conv2d_float32(uint16_t input_depth, uint16_t input_height, uint16_t input_width, float* input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, float* output,
const float* bias,
uint16_t kernel_height, uint16_t kernel_width, const float* kernel,
uint8_t relu){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1

    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                output[out_d * output_height * output_width + out_h * output_width + out_w] = 0;
                for(uint16_t in_d = 0; in_d < input_depth; in_d++){
                    output[out_d * output_height * output_width + out_h * output_width + out_w] += 
                        input[in_d * input_height * input_width + out_h * input_width + out_w] * 
                            kernel[out_d * input_depth + in_d];
                }

                output[out_d * output_height * output_width + out_h * output_width + out_w] += bias[out_d];

                if(relu == 1){
                    if(output[out_d * output_height * output_width + out_h * output_width + out_w] < 0){
                        output[out_d * output_height * output_width + out_h * output_width + out_w] = 0;
                    }
                }
            }
        }
    }
    return(0);
}
