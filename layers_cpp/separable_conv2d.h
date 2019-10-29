#include <stdint.h>

uint8_t separable_conv2d_fix16(uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< int16_t> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< int16_t> > >& output,
const vector< int16_t >& bias_d, const vector< int16_t >& bias_p,
uint16_t kernel_d_height, uint16_t kernel_d_width, const vector< vector< vector< vector< int16_t> > > >& kernel_d, const vector< vector< vector< vector< int16_t> > > >& kernel_p,
uint8_t relu, uint8_t fractal_width);

uint8_t separable_conv2d_float32(uint16_t input_depth, uint16_t input_height, uint16_t input_width, vector< vector< vector< float> > >& input,
uint16_t output_depth, uint16_t output_height, uint16_t output_width, vector< vector< vector< float> > >& output,
const vector< float >& bias_d, const vector< float >& bias_p,
uint16_t kernel_d_height, uint16_t kernel_d_width, const vector< vector< vector< vector< float> > > >& kernel_d, const vector< vector< vector< vector< float> > > >& kernel_p,
uint8_t relu);