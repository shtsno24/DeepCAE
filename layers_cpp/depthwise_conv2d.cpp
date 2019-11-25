#include <cstdint>
#include <vector>
#include "depthwise_conv2d.h"

using namespace std;

uint8_t depthwise_conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output,
const vector< int16_t >& bias,
uint16_t kernel_height, uint16_t kernel_width, const vector< vector< vector< vector< int16_t> > > >& kernel,
uint8_t relu, uint8_t fractal_width){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1

    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                output[out_d][out_h][out_w] = 0;
                for(uint16_t k_h = 0; k_h < kernel_height; k_h++){
                    for(uint16_t k_w = 0; k_w < kernel_width; k_w++){
                        output[out_d][out_h][out_w] += 
                            (int16_t)(((int32_t)(input[out_d][out_h + k_h][out_w + k_w]) * (int32_t)(kernel[0][out_d][k_h][k_w]))
                            >> fractal_width);
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

uint8_t depthwise_conv2d_float32(uint16_t input_depth, uint16_t input_height, uint16_t input_width,  vector< vector< vector< float> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output,
const vector< float >& bias,
uint16_t kernel_height, uint16_t kernel_width, const vector< vector< vector< vector< float> > > >& kernel,
uint8_t relu){
    // uint16_t output_hight = (input_shape.height + 2 * pad - kernel_shape.height) / stride + 1;
    // uint16_t output_width = (input_shape.width + 2 * pad - kernel_shape.width) / stride + 1;
    
    // input_size *must* be included padding size
    // stride is fixed to 1


    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                output[out_d][out_h][out_w] = 0;
                for(uint16_t k_h = 0; k_h < kernel_height; k_h++){
                    for(uint16_t k_w = 0; k_w < kernel_width; k_w++){
                        output[out_d][out_h][out_w] += 
                            input[out_d][out_h + k_h][out_w + k_w] * 
                            kernel[0][out_d][k_h][k_w];
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
