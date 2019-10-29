#include <cstdint>
#include <vector>

#include "conv2d.h"
#include "depthwise_conv2d.h"
#include "separable_conv2d.h"

using namespace std;

uint8_t separable_conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output,
const vector< int16_t >& bias_d, const vector< int16_t >& bias_p,
uint16_t kernel_d_height, uint16_t kernel_d_width, const vector< vector< vector< vector< int16_t> > > >& kernel_d, const vector< vector< vector< vector< int16_t> > > >& kernel_p,
uint8_t relu, uint8_t fractal_width){ 

    vector< vector< vector< int16_t> > > middle_array(input_depth, vector< vector < int16_t> >(output_height, vector< int16_t>(output_width)));

    depthwise_conv2d_fix16(input_depth, input_height, input_depth, input,
                            input_depth, output_height, output_width, middle_array,
                            bias_d,
                            kernel_d_height, kernel_d_width, kernel_d, relu, fractal_width);

    conv2d_fix16(input_depth, output_height, output_width, middle_array,
                output_depth, output_height, output_width, output,
                bias_p,
                1, 1, kernel_p, relu, fractal_width);

    return (0);
}

uint8_t separable_conv2d_float32(uint16_t input_depth, uint16_t input_height, vector< vector< vector< float> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output,
const vector< float >& bias_d, const vector< float >& bias_p,
uint16_t kernel_d_height, uint16_t kernel_d_width, const vector< vector< vector< vector< float> > > >& kernel_d, const vector< vector< vector< vector< float> > > >& kernel_p,
uint8_t relu){

    vector< vector< vector< float> > > middle_array(input_depth, vector< vector < float> >(output_height, vector< float>(output_width)));

    depthwise_conv2d_float32(input_depth, input_height, input_depth, input,
                            input_depth, output_height, output_width, middle_array,
                            bias_d,
                            kernel_d_height, kernel_d_width, kernel_d, relu);

    conv2d_float32(input_depth, output_height, output_width, middle_array,
                    output_depth, output_height, output_width, output,
                    bias_p,
                    1, 1, kernel_p, relu);

    return (0);
}