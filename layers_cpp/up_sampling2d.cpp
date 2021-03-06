#include <cstdint>
#include <vector>
#include "up_sampling2d.h"

using namespace std;

uint8_t up_sampling2d_fix16(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output){

    // output_* "must" be Divisible by kernel_size on any axis
    
    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                            output[out_d][out_h][out_w] = input[out_d][out_h/kernel_size][out_w/kernel_size];
            }
        }
    }
    return(0);
}

uint8_t up_sampling2d_float32(uint16_t kernel_size,
uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< float> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output){

    // output_* "must" be Divisible by kernel_size on any axis
    
    for(uint16_t out_d = 0; out_d < output_depth; out_d++){
        for(uint16_t out_h = 0; out_h < output_height; out_h++){
            for(uint16_t out_w = 0; out_w < output_width; out_w++){
                            output[out_d][out_h][out_w] = input[out_d][out_h/kernel_size][out_w/kernel_size];
            }
        }
    }
    return(0);
}